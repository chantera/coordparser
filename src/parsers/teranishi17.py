import chainer
import chainer.functions as F
import numpy as np

from dataset import Coordination
from models.common import get_pair_value
from models.teranishi17 import Teranishi17
from parsers.common import Parser


class Teranishi17Parser(Parser):

    def __init__(self, model, comma_id, decoding=True):
        if not isinstance(model, Teranishi17):
            raise ValueError("Unsupported model: {}"
                             .format(model.__class__.__name__))
        assert comma_id is not None
        self.model = model
        self.comma_id = comma_id
        self.decoding = decoding

    def parse(self, words, postags, chars, cc_indices, sep_indices,
              cont_embeds=None, n_best=1, use_cache=False):
        if n_best != 1:
            raise ValueError("Only supported 1 best parsing")
        coords_batch = []
        if not use_cache or not self.model.has_cache():
            self.model.forward(words, postags, chars,
                               cc_indices, sep_indices, cont_embeds)
        parsed = self.model.cache
        pairs = parsed['pairs']
        ckey_scores = chainer.cuda.to_cpu(parsed['ckey_scores'].data)
        pair_scores = chainer.cuda.to_cpu(parsed['pair_scores'].data)
        ckey_scores = F.split_axis(
            ckey_scores, parsed['ckey_offsets'][:-1], axis=0)
        pair_scores = F.split_axis(
            pair_scores, parsed['pair_offsets'][:-1], axis=0)
        coord_index = 0
        for batch_index in range(len(words)):
            n_ccs = len(cc_indices[batch_index])
            if n_ccs == 0:
                coords_batch.append([])
                continue
            results = self._parse_each(words[batch_index],
                                       cc_indices[batch_index],
                                       pairs[batch_index],
                                       ckey_scores[batch_index],
                                       pair_scores[coord_index:
                                                   coord_index + n_ccs],
                                       n_best)
            coords_batch.append(results)
            coord_index += n_ccs
        return coords_batch

    def _parse_each(self, words, cc_indices, pairs,
                    cc_scores, pair_scores, n_best):
        pairs = [[(idx1, idx2) for idx1, idx2 in zip(*get_pair_value(idx))]
                 if idx is not None else None for idx in pairs]
        if self.decoding:
            coords, score = _solve_coords(
                words, cc_indices, pairs,
                cc_scores.data, pair_scores, self.comma_id)
        else:
            coords, score = _solve_coords_independently(
                words, cc_indices, pairs, cc_scores.data, pair_scores,
                self.comma_id)
        coord_dict = {}
        if coords is None:
            coords = [None] * len(cc_indices)
        for conjuncts, cc in zip(coords, cc_indices):
            if conjuncts is not None:
                coord = Coordination(cc, conjuncts)
            else:
                coord = None
            coord_dict[cc] = coord
        return [(coord_dict, score)]


def _solve_coords_independently(words, cc_indices, pairs_all,
                                cc_scores, pair_scores_all, comma_id):
    coords_conjuncts = []
    score = 0.0
    seps = [i for i, word in enumerate(words) if word == comma_id]
    for i, cc in enumerate(cc_indices):
        scores = F.concat((cc_scores[i], pair_scores_all[i]), axis=0).data
        idx = F.argmax(scores, axis=0).data
        if idx > 0:
            pair = pairs_all[i][idx - 1]
            conjuncts = _split_coord(pair, cc, seps)
        else:
            conjuncts = None
        coords_conjuncts.append(conjuncts)
        score += scores[idx]
    return (coords_conjuncts, score)


def _solve_coords(words, cc_indices, pairs_all, cc_scores, pair_scores_all,
                  comma_id, filter_nested_in_conj=False):
    MAX_AGENDA_SIZE = 1024
    agenda = []
    seps = [i for i, word in enumerate(words) if word == comma_id]
    for cc, pairs, cc_score, pair_scores in \
            zip(cc_indices, pairs_all, cc_scores, pair_scores_all):
        if isinstance(cc_score, chainer.Variable):
            cc_score = cc_score.data
        if isinstance(pair_scores, chainer.Variable):
            pair_scores = pair_scores.data
        if pairs is not None:
            pairs = [None] + pairs
            scores = np.concatenate((cc_score, pair_scores), axis=0)
        else:
            pairs = [None]
            scores = cc_score
        assert len(pairs) == scores.shape[0]
        if len(agenda) == 0:
            agenda = [[[pair], score] for pair, score in zip(pairs, scores)]
            agenda.sort(key=lambda x: x[1], reverse=True)
            del agenda[MAX_AGENDA_SIZE:]
            continue
        new_agenda = []
        for candidate_pair, candidate_score in zip(pairs, scores):
            for pair_comb, score_comb in agenda:
                is_valid = all(is_valid_coords(pair, candidate_pair)
                               for pair in pair_comb)
                if is_valid:
                    new_agenda.append([pair_comb + [candidate_pair],
                                       score_comb + candidate_score])
        agenda = new_agenda
        agenda.sort(key=lambda x: x[1], reverse=True)
        del agenda[MAX_AGENDA_SIZE:]
    for pairs, score in agenda:
        coords_conjuncts = [_split_coord(pair, cc, seps)
                            if pair is not None else None
                            for pair, cc in zip(pairs, cc_indices)]
        is_valid = True
        if filter_nested_in_conj:
            is_valid = True
            for i, next_conjuncts in enumerate(coords_conjuncts):
                is_valid = all(is_valid_conjuncts(conjuncts, next_conjuncts)
                               for conjuncts in coords_conjuncts[:i])
                if not is_valid:
                    break
        if is_valid:
            return (coords_conjuncts, score)
    return (None, -np.inf)


def _split_coord(coord, cc, seps):
    spans = []
    buf = []
    for i in range(coord[0], cc):
        if i not in seps:
            buf.append(i)
        elif len(buf) > 0:
            spans.append((buf[0], buf[-1]))
            buf = []
    if len(buf) > 0:
        spans.append((buf[0], buf[-1]))
    if len(spans) == 0:
        spans.append((coord[0], cc - 1))
    right_begin = (cc + 1 if cc + 1 not in seps else cc + 2)
    spans.append((right_begin, coord[1]))
    assert len(spans) >= 2
    return spans


def is_valid_coords(a, b):
    return a is None \
        or b is None \
        or a[1] < b[0] \
        or b[1] < a[0] \
        or (a[0] <= b[0] and b[1] <= a[1]) \
        or (b[0] <= a[0] and a[1] <= b[1])


def is_valid_conjuncts(a, b):
    if a is None or b is None:
        return None
    if a[-1][1] < b[0][0] or b[-1][1] < a[0][0]:
        return True
    if a[0][0] <= b[0][0] and b[-1][1] <= a[-1][1]:
        b_coord = (b[0][0], b[-1][1])
        return any(a_span[0] <= b_coord[0] and b_coord[1] <= a_span[1]
                   for a_span in a)
    if b[0][0] <= a[0][0] and a[-1][1] <= b[-1][1]:
        a_coord = (a[0][0], a[-1][1])
        return any(b_span[0] <= a_coord[0] and a_coord[1] <= b_span[1]
                   for b_span in b)
    return False
