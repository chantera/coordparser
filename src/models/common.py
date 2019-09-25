from enum import IntEnum

import chainer
import chainer.functions as F
import chainer_nn.functions as nn_F
import chainer_nn.links as nn_L
import numpy as np


MAX_SENTENCE_LENGTH = 512

_pair_dict = np.arange(MAX_SENTENCE_LENGTH ** 2, dtype=np.int32) \
    .reshape(MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH)

_pair_kv = np.vstack(np.where(_pair_dict >= 0)).T


def get_pair_key(index1, index2):
    return _pair_dict[index1, index2]


def get_pair_value(key):
    idx1, idx2 = _pair_kv[key].T
    return idx1, idx2


class CKeyType(IntEnum):
    CC = 0
    SEP = 1


class CoordSolver(chainer.Chain):

    def __init__(self, encoder, ckey_classifier, use_separators=True):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.ckey_clf = ckey_classifier
        self._cache = {}
        self._use_seps = use_separators

    def forward(self, words, postags, chars, cc_indices, sep_indices,
                cont_embeds, force_compute_scores=False):
        self.clear_cache()
        hs, lengths = self.encoder(words, postags, chars, cont_embeds)
        assert np.all(lengths <= MAX_SENTENCE_LENGTH)
        if not isinstance(hs, (tuple, list)):
            hs = nn_F.unpad_sequence(hs, lengths)
        if not self._use_seps:
            sep_indices = None
        ckeys, ckey_types = _merge_cc_and_sep(cc_indices, sep_indices)
        self._cache.update({
            'lengths': lengths,
            'encoded_seqs': hs,
            'ckeys': ckeys,
            'ckey_types': ckey_types,
        })
        if force_compute_scores or not chainer.config.train:
            self._cache.update(
                self._forward_scores(hs, lengths, ckeys, ckey_types))
        return self._cache

    def _forward_scores(self, hs, lengths, ckeys, ckey_types):
        raise NotImplementedError

    def compute_loss(self, output, gold):
        raise NotImplementedError

    def compute_accuracy(self, output, gold):
        raise NotImplementedError

    def clear_cache(self):
        self._cache.clear()
        self._cache.update({
            'lengths': None,
            'encoded_seqs': None,
            'ckeys': None,
            'ckey_types': None,
        })

    def has_cache(self):
        return any(v is not None for v in self._cache.values())

    @property
    def cache(self):
        return self._cache

    @property
    def result(self):
        return self.cache


def _merge_cc_and_sep(cc_indices, sep_indices):
    ckeys = list(cc_indices)
    ckey_types = [np.full_like(idxs_i, CKeyType.CC)
                  for idxs_i in cc_indices]
    if sep_indices is not None:
        assert len(ckeys) == len(sep_indices)
        for i in range(len(ckeys)):
            if len(ckeys[i]) == 0:
                continue
            ckeys[i] = np.concatenate((ckeys[i], sep_indices[i]))
            ckey_types[i] = np.concatenate(
                (ckey_types[i], np.full_like(
                    sep_indices[i], CKeyType.SEP)))
    return ckeys, ckey_types


_glorotnormal = chainer.initializers.GlorotNormal()


class CKeyClassifier(chainer.Chain):

    def __init__(self, in_size, out_size=1):
        super().__init__()
        with self.init_scope():
            self.linear = chainer.links.Linear(
                in_size, out_size, initialW=_glorotnormal)
        self._out_size = out_size

    def forward(self, ckeys, hs_flatten, lengths):
        n_ckeys = np.array([len(ckeys_i) for ckeys_i in ckeys], np.int32)
        ckeys = [ckeys_i + offset for ckeys_i, offset
                 in zip(ckeys, np.insert(lengths, 0, 0)[:-1].cumsum())]
        ckeys = np.concatenate(ckeys).astype(np.int32)
        hs_ckeys = F.embed_id(self.xp.asarray(ckeys), hs_flatten)
        scores = self.linear(hs_ckeys)
        return scores, n_ckeys.cumsum().astype(np.int32)

    @property
    def out_size(self):
        return self._out_size


class PairScorer(chainer.Chain):

    def __init__(self, extractor, mlp_unit=None, dropout=0.0):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor
            if mlp_unit is None:
                mlp_unit = extractor.out_size // 2
            self.mlp = nn_L.MLP([
                nn_L.MLP.Layer(extractor.out_size, mlp_unit, F.relu,
                               dropout, initialW=_glorotnormal),
                nn_L.MLP.Layer(mlp_unit, 1, initialW=_glorotnormal)])

    def forward(self, pairs, ckeys, hs_flatten, lengths):
        pairs, ckeys, offsets \
            = self._flatten_ckeys_and_pairs(pairs, ckeys, lengths)
        if len(pairs) == 0:
            scores = chainer.Variable(self.xp.empty((0,), self.xp.float32))
            return scores, offsets
        features = self.extractor(hs_flatten, pairs, ckeys, lengths)
        scores = F.squeeze(self.mlp(features), axis=1)
        return scores, offsets

    @staticmethod
    def _flatten_ckeys_and_pairs(pairs, ckeys, lengths):
        n_pairs = np.array([len(pairs_j) for pairs_i in pairs
                            for pairs_j in pairs_i], np.int32)
        idxs1, idxs2, ckeys = zip(*[
            [idxs + offset for idxs
             in get_pair_value(pairs_j) + (np.tile(ckey_j, len(pairs_j)),)]
            for pairs_i, ckeys_i, offset in zip(
                pairs, ckeys, np.insert(lengths, 0, 0)[:-1].cumsum())
            for pairs_j, ckey_j in zip(pairs_i, ckeys_i)])
        idxs1 = np.concatenate(idxs1).astype(np.int32)
        idxs2 = np.concatenate(idxs2).astype(np.int32)
        pairs = np.vstack((idxs1, idxs2)).T
        ckeys = np.concatenate(ckeys).astype(np.int32)
        return pairs, ckeys, n_pairs.cumsum().astype(np.int32)


class FeatureExtractor(chainer.Chain):

    def forward(self, hs_flatten, pairs, ckeys, lengths):
        raise NotImplementedError

    @property
    def out_size(self):
        raise NotImplementedError


def accuracy(y, t, ignore_label=None):
    if isinstance(y, chainer.Variable):
        y = y.data
    if isinstance(t, chainer.Variable):
        t = t.data
    xp = chainer.cuda.get_array_module(y)
    pred = y.argmax(axis=1).reshape(t.shape)
    if ignore_label is not None:
        mask = (t == ignore_label)
        ignore_cnt = mask.sum()
        pred = xp.where(mask, ignore_label, pred)
        count = (pred == t).sum() - ignore_cnt
        total = t.size - ignore_cnt
    else:
        count = (pred == t).sum()
        total = t.size
    return count, total
