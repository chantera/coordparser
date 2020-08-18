import chainer
import chainer.functions as F
import numpy as np

from models import common
from models.common import get_pair_key


class Teranishi17(common.CoordSolver):
    _CACHE_ENTRIES = (
        'lengths',
        'encoded_seqs',
        'ckeys',
        'ckey_types',
        'ckey_scores',
        'ckey_offsets',
        'pairs',
        'pair_scores',
        'pair_offsets',
        'coord_loss',
        'coord_accuracy',
    )

    def __init__(self, encoder, ckey_classifier, pair_scorer):
        assert ckey_classifier.out_size == 1
        super().__init__(encoder, ckey_classifier, use_separators=False)
        with self.init_scope():
            self.pair_scorer = pair_scorer

    def _forward_scores(self, hs, lengths, ckeys, ckey_types,
                        gold_pairs=None):
        hs_flatten = F.concat(hs, axis=0)
        ckey_scores, ckey_offsets = self.ckey_clf(ckeys, hs_flatten, lengths)
        pairs = _enumerate_pairs_batch(lengths, ckeys, ckey_types, gold_pairs)
        scores, offsets = self.pair_scorer(
            [[pairs_j if pairs_j is not None else (np.array([], np.int32))
              for pairs_j in pairs_i] for pairs_i in pairs],
            ckeys, hs_flatten, lengths)
        output = {
            'ckey_scores': ckey_scores,
            'ckey_offsets': ckey_offsets,
            'pairs': pairs,
            'pair_scores': scores,
            'pair_offsets': offsets,
        }
        return output

    def compute_loss(self, output, gold):
        self._cache.update(self.compute_metrics(output, gold))
        return self._cache['coord_loss']

    def compute_accuracy(self, output, gold, use_cache=True):
        if not use_cache or self._cache.get('coord_accuracy') is None:
            self._cache.update(self.compute_metrics(output, gold))
        return self._cache['coord_accuracy']

    def compute_metrics(self, output, gold):
        gold_pairs = [_get_true_pair(ckey, ckey_type, coords_i)
                      for ckeys_i, ckey_types_i, coords_i
                      in zip(output['ckeys'], output['ckey_types'], gold)
                      for ckey, ckey_type in zip(ckeys_i, ckey_types_i)]
        if output.get('ckey_scores') is None:
            output.update(self._forward_scores(
                output['encoded_seqs'], output['lengths'],
                output['ckeys'], output['ckey_types'], gold_pairs))
        ckey_scores = output['ckey_scores']
        xp = chainer.cuda.get_array_module(ckey_scores)

        scores = output['pair_scores']
        offsets = output['pair_offsets']
        scores = list(F.split_axis(scores, offsets[:-1], axis=0))
        indices = []
        k = 0
        for pairs_i in output['pairs']:
            for pairs_j in pairs_i:
                gold_pair = gold_pairs[k]
                if pairs_j is None or gold_pair is None:
                    indices.append(-1)
                else:
                    idx = np.argwhere(pairs_j == get_pair_key(*gold_pair))
                    assert idx.size == 1
                    idx = idx[0, 0]
                    indices.append(idx)
                k += 1

        indices = xp.asarray(indices, xp.int32)
        if offsets[-1] > 0:
            scores = F.pad_sequence(scores, padding=-np.inf)
            assert scores.shape[0] == indices.size
            scores = F.hstack((ckey_scores, scores))
        else:
            scores = ckey_scores
        indices += 1
        coord_loss = F.softmax_cross_entropy(scores, indices)
        coord_accuracy = common.accuracy(scores, indices)

        result = {'coord_loss': coord_loss, 'coord_accuracy': coord_accuracy}
        return result

    def clear_cache(self):
        super().clear_cache()
        self._cache.update({k: None for k in self._CACHE_ENTRIES})


def _enumerate_pairs_batch(lengths, ckeys, ckey_types, gold_pairs=None):
    pairs_batch = []
    k = 0
    for i, (length, ckeys_i, ckey_types_i) \
            in enumerate(zip(lengths, ckeys, ckey_types)):
        pairs_in_seq = []
        assert any(ckey_type == common.CKeyType.CC
                   for ckey_type in ckey_types_i)
        for j, (ckey, ckey_type) in enumerate(zip(ckeys_i, ckey_types_i)):
            if gold_pairs is not None and gold_pairs[k] is None:
                pairs = None
            else:
                pairs = _enumerate_pairs(length, ckey)
            pairs_in_seq.append(pairs)
            k += 1
        pairs_batch.append(pairs_in_seq)
    return pairs_batch


def _enumerate_pairs(n, ckey):
    if ckey == 0 or ckey == n - 1:
        pairs = None
    else:
        pairs = get_pair_key(
            np.tile(np.arange(0, ckey), (n - ckey - 1, 1)).T.reshape(-1),
            np.tile(np.arange(ckey + 1, n), ckey))
        return pairs
    return pairs


def _get_true_pair(ckey, ckey_type, coords):
    span = None
    if ckey_type == common.CKeyType.CC:
        if coords[ckey] is not None:
            conjuncts = coords[ckey].conjuncts
            span = (conjuncts[0][0], conjuncts[-1][1])
    elif ckey_type == 1:
        raise ValueError("unsupported ckey type: {}".format(ckey_type))
    else:
        raise ValueError("unknown ckey type: {}".format(ckey_type))
    return span
