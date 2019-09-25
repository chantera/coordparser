from enum import Enum

import chainer
import chainer.functions as F
import numpy as np

from models import common
from models.common import get_pair_key


class PairDivision(Enum):
    LEFT_RIGHT = 0
    BEGIN_END = 1
    INNER_OUTER = 2


class Teranishi19(common.CoordSolver):
    _CACHE_ENTRIES = (
        'lengths',
        'encoded_seqs',
        'ckeys',
        'ckey_types',
        'ckey_scores',
        'ckey_offsets',
        'bipairs',
        'bipair_scores',
        'bipair_offsets',
        'ckey_loss',
        'ckey_accuracy',
        'pair1_loss',
        'pair1_accuracy',
        'pair2_loss',
        'pair2_accuracy',
    )

    def __init__(self, encoder, ckey_classifier, pair_scorer1, pair_scorer2,
                 division='inner_outer'):
        assert ckey_classifier.out_size == 2
        if division == 'left_right' or division == PairDivision.LEFT_RIGHT:
            division = PairDivision.LEFT_RIGHT
        elif division == 'begin_end' or division == PairDivision.BEGIN_END:
            division = PairDivision.BEGIN_END
        elif division == 'inner_outer' or division == PairDivision.INNER_OUTER:
            division = PairDivision.INNER_OUTER
        else:
            raise ValueError("unknown division: {}".format(division))
        super().__init__(encoder, ckey_classifier, use_separators=True)
        self._division = division
        with self.init_scope():
            self.pair_scorer1 = pair_scorer1
            self.pair_scorer2 = pair_scorer2
        self._score_table = ScoreTable(self._division)

    def forward(self, words, postags, chars, cc_indices, sep_indices,
                cont_embeds, force_compute_scores=False):
        self._score_table.clear()
        output = super().forward(
            words, postags, chars, cc_indices, sep_indices, cont_embeds,
            force_compute_scores)
        if self._cache['bipair_scores'] is not None:
            self._update_score_table(self._cache)
        return output

    def _forward_scores(self, hs, lengths, ckeys, ckey_types,
                        gold_bipairs=None):
        hs_flatten = F.concat(hs, axis=0)
        ckey_scores, ckey_offsets = self.ckey_clf(ckeys, hs_flatten, lengths)
        bipairs = _enumerate_bipairs_batch(
            lengths, ckeys, ckey_types, self._division, gold_bipairs)
        pairs1, pairs2 = zip(*[tuple(zip(*[
            bipairs_j if bipairs_j is not None
            else (np.array([], np.int32), np.array([], np.int32))
            for bipairs_j in bipairs_i])) for bipairs_i in bipairs])
        scores1, offsets1 \
            = self.pair_scorer1(pairs1, ckeys, hs_flatten, lengths)
        scores2, offsets2 \
            = self.pair_scorer2(pairs2, ckeys, hs_flatten, lengths)
        output = {
            'ckey_scores': ckey_scores,
            'ckey_offsets': ckey_offsets,
            'bipairs': bipairs,
            'bipair_scores': (scores1, scores2),
            'bipair_offsets': (offsets1, offsets2),
        }
        return output

    def compute_loss(self, output, gold):
        self._cache.update(self.compute_metrics(output, gold))
        return self._cache['ckey_loss'] \
            + self._cache['pair1_loss'] + self._cache['pair2_loss']

    def compute_accuracy(self, output, gold, use_cache=True):
        if not use_cache or self._cache.get('ckey_accuracy') is None:
            self._cache.update(self.compute_metrics(output, gold))
        return (self._cache['ckey_accuracy'],
                self._cache['pair1_accuracy'], self._cache['pair2_accuracy'])

    def compute_metrics(self, output, gold):
        division = self._division
        ckey_labels, gold_bipairs = zip(*[
            (_get_true_ckey_label(ckey, ckey_type, coords_i),
             _get_true_bipair(ckey, ckey_type, coords_i, division))
            for ckeys_i, ckey_types_i, coords_i
            in zip(output['ckeys'], output['ckey_types'], gold)
            for ckey, ckey_type in zip(ckeys_i, ckey_types_i)])
        if output.get('ckey_scores') is None:
            output.update(self._forward_scores(
                output['encoded_seqs'], output['lengths'],
                output['ckeys'], output['ckey_types'], gold_bipairs))
        ckey_scores = output['ckey_scores']
        xp = chainer.cuda.get_array_module(ckey_scores)
        ckey_labels = xp.asarray(ckey_labels, xp.int32)
        ckey_loss = F.softmax_cross_entropy(ckey_scores, ckey_labels)
        ckey_accuracy = common.accuracy(ckey_scores, ckey_labels)

        scores1, scores2 = output['bipair_scores']
        offsets1, offsets2 = output['bipair_offsets']
        scores1 = list(F.split_axis(scores1, offsets1[:-1], axis=0))
        scores2 = list(F.split_axis(scores2, offsets2[:-1], axis=0))
        indices1, indices2 = [], []
        exclude = []
        k = 0
        for bipairs_i in output['bipairs']:
            for bipairs_j in bipairs_i:
                gold_bipair = gold_bipairs[k]
                if bipairs_j is None or gold_bipair is None:
                    exclude.append(k)
                else:
                    idx1 = np.argwhere(
                        bipairs_j[0] == get_pair_key(*gold_bipair[0]))
                    idx2 = np.argwhere(
                        bipairs_j[1] == get_pair_key(*gold_bipair[1]))
                    assert idx1.size == idx2.size == 1
                    idx1, idx2 = idx1[0, 0], idx2[0, 0]
                    indices1.append(idx1)
                    indices2.append(idx2)
                k += 1
        for k in reversed(exclude):
            del scores1[k], scores2[k]

        if len(scores1) > 0:
            scores1 = F.pad_sequence(scores1, padding=-np.inf)
            indices1 = xp.asarray(indices1, xp.int32)
            assert scores1.shape[0] == indices1.size
            pair_loss1 \
                = F.softmax_cross_entropy(scores1, indices1, reduce='no')
            pair_loss1 = F.sum(pair_loss1) / ckey_labels.size
            pair_accuracy1 = common.accuracy(scores1, indices1)
        else:
            pair_loss1 = chainer.Variable(xp.array(0.0, xp.float32))
            pair_accuracy1 = (0, 0)

        if len(scores2) > 0:
            scores2 = F.pad_sequence(scores2, padding=-np.inf)
            indices2 = xp.asarray(indices2, xp.int32)
            assert scores2.shape[0] == indices2.size
            pair_loss2 \
                = F.softmax_cross_entropy(scores2, indices2, reduce='no')
            pair_loss2 = F.sum(pair_loss2) / ckey_labels.size
            pair_accuracy2 = common.accuracy(scores2, indices2)
        else:
            pair_loss2 = chainer.Variable(xp.array(0.0, xp.float32))
            pair_accuracy2 = (0, 0)

        result = {
            'ckey_loss': ckey_loss, 'ckey_accuracy': ckey_accuracy,
            'pair1_loss': pair_loss1, 'pair1_accuracy': pair_accuracy1,
            'pair2_loss': pair_loss2, 'pair2_accuracy': pair_accuracy2,
        }
        return result

    def _update_score_table(self, output):
        to_cpu = chainer.cuda.to_cpu
        ckey_scores = output['ckey_scores']
        ckey_scores = F.log_softmax(output['ckey_scores'], axis=1)
        ckey_scores = F.split_axis(
            to_cpu(ckey_scores.data), output['ckey_offsets'][:-1], axis=0)

        scores1, scores2 = output['bipair_scores']
        offsets1, offsets2 = output['bipair_offsets']
        scores1 = F.split_axis(to_cpu(scores1.data), offsets1[:-1], axis=0)
        scores2 = F.split_axis(to_cpu(scores2.data), offsets2[:-1], axis=0)

        ckeys, bipairs = output['ckeys'], output['bipairs']
        ckey_scores_batch = []
        bipair_scores_batch = []
        k = 0
        for i, ckeys_i in enumerate(ckeys):
            ckey_scores_batch.append(ckey_scores[i].data)
            bipair_scores_i = []
            for ckey_j in ckeys_i:
                scores1_j = scores1[k]
                if scores1_j.size > 0:
                    scores1_j = F.log_softmax(scores1_j, axis=0)
                scores2_j = scores2[k]
                if scores2_j.size > 0:
                    scores2_j = F.log_softmax(scores2_j, axis=0)
                bipair_scores_i.append((scores1_j.data, scores2_j.data))
                k += 1
            bipair_scores_batch.append(bipair_scores_i)

        self._score_table.set_scores(
            ckeys, ckey_scores_batch, bipairs, bipair_scores_batch)

    def clear_cache(self):
        super().clear_cache()
        self._cache.update({k: None for k in self._CACHE_ENTRIES})

    @property
    def score_table(self):
        return self._score_table


class ScoreTable(object):

    def __init__(self, division):
        self._division = division
        self._data = []

    def clear(self):
        self._data.clear()

    def set_scores(self, ckeys, ckey_scores, bipairs, bipair_scores):
        assert (len(ckeys) == len(ckey_scores)
                == len(bipairs) == len(bipair_scores))
        for ckeys_i, ckey_scores_i, bipairs_i, bipair_scores_i \
                in zip(ckeys, ckey_scores, bipairs, bipair_scores):
            assert (len(ckeys_i) == len(ckey_scores_i)
                    == len(bipairs_i) == len(bipair_scores_i))
            entries = {}
            for ckey_j, ckey_score_j, bipairs_j, bipair_scores_j \
                    in zip(ckeys_i, ckey_scores_i, bipairs_i, bipair_scores_i):
                entry = {
                    'ckey_score': ckey_score_j,
                    'bipairs': bipairs_j,
                    'bipair_scores': bipair_scores_j,
                    'pairs1_entries': None,
                    'pairs2_entries': None,
                }
                if bipairs_j is not None:
                    pairs1, pairs2 = bipairs_j
                    scores1, scores2 = bipair_scores_j
                    entries1 = {k: v for k, v in zip(pairs1, scores1)}
                    entries2 = {k: v for k, v in zip(pairs2, scores2)}
                    entry['pairs1_entries'] = entries1
                    entry['pairs2_entries'] = entries2
                entries[ckey_j] = entry
            self._data.append(entries)

    def get_entries(self, sentence_index):
        return self._data[sentence_index]

    def lookup_bispan_score(self, sentence_index, ckey, bispan):
        entry = self.get_entries(sentence_index)[ckey]
        bipair = _bispan_to_bipair(bispan, self._division)
        score1 = entry['pairs1_entries'][get_pair_key(*bipair[0])]
        score2 = entry['pairs2_entries'][get_pair_key(*bipair[1])]
        return score1 + score2


def _enumerate_bipairs_batch(lengths, ckeys, ckey_types, division,
                             gold_bipairs=None):
    bipairs_batch = []
    k = 0
    for i, (length, ckeys_i, ckey_types_i) \
            in enumerate(zip(lengths, ckeys, ckey_types)):
        bipairs_in_seq = []
        assert any(ckey_type == common.CKeyType.CC
                   for ckey_type in ckey_types_i)
        for j, (ckey, ckey_type) in enumerate(zip(ckeys_i, ckey_types_i)):
            if gold_bipairs is not None and gold_bipairs[k] is None:
                bipairs = None
            else:
                bipairs = _enumerate_bipairs(length, ckey, division)
            bipairs_in_seq.append(bipairs)
            k += 1
        bipairs_batch.append(bipairs_in_seq)
    return bipairs_batch


def _enumerate_bipairs(n, ckey, division):
    if division == PairDivision.LEFT_RIGHT:
        raise NotImplementedError
    elif (division == PairDivision.BEGIN_END
          or division == PairDivision.INNER_OUTER):
        if ckey == 0 or ckey == n - 1:
            bipairs = None
        else:
            pairs = get_pair_key(
                np.tile(np.arange(0, ckey), (n - ckey - 1, 1)).T.reshape(-1),
                np.tile(np.arange(ckey + 1, n), ckey))
            bipairs = (pairs, pairs)
    else:
        raise ValueError('unknown division: {}'.format(division))
    return bipairs


def _get_true_ckey_label(ckey, ckey_type, coords):
    if ckey_type == common.CKeyType.CC:
        label = int(coords[ckey] is not None)
    elif ckey_type == common.CKeyType.SEP:
        label = int(any(coord is not None and ckey in coord.seps
                        for coord in coords.values()))
    else:
        raise ValueError("unknown ckey type: {}".format(ckey_type))
    return label


def _get_true_bipair(ckey, ckey_type, coords, division):
    bispan = None
    if ckey_type == common.CKeyType.CC:
        if coords[ckey] is not None:
            bispan = coords[ckey].get_pair(ckey, check=True)
    elif ckey_type == 1:
        for coord in coords.values():
            if coord is not None and ckey in coord.seps:
                bispan = coord.get_pair(ckey, check=True)
                break
    else:
        raise ValueError("unknown ckey type: {}".format(ckey_type))
    return _bispan_to_bipair(bispan, division)


def _bispan_to_bipair(bispan, division):
    bipair = bispan
    if division == PairDivision.LEFT_RIGHT:
        pass
    elif division == PairDivision.BEGIN_END:
        if bipair is not None:
            pair1, pair2 = bipair
            bipair = ((pair1[0], pair2[0]), (pair1[1], pair2[1]))
    elif division == PairDivision.INNER_OUTER:
        if bipair is not None:
            pair1, pair2 = bipair
            bipair = ((pair1[1], pair2[0]), (pair1[0], pair2[1]))
    else:
        raise ValueError('unknown division: {}'.format(division))
    return bipair
