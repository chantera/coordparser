import numpy as np

from models import common


class GoldModel(common.CoordSolver):

    def __init__(self):
        super().__init__(None, None)
        self._score_table = ScoreTable()
        self._gold = None

    def set_gold(self, coords):
        self._gold = coords

    def forward(self, words, postags, chars, cc_indices, sep_indices,
                cont_embeds, force_compute_scores=False):
        self.clear_cache()
        self._score_table.clear()
        self._score_table.set_scores(self._gold, cc_indices, sep_indices)
        return {}

    def _forward_scores(self, hs, lengths, ckeys, ckey_types):
        raise NotImplementedError

    def compute_loss(self, output, gold):
        raise NotImplementedError

    def compute_accuracy(self, output, gold):
        raise NotImplementedError

    @property
    def score_table(self):
        return self._score_table


class ScoreTable(object):

    def __init__(self):
        self._data = []

    def clear(self):
        self._data.clear()

    def set_scores(self, coords, cc_indices, sep_indices):
        for coords_i, cc_indices_i, sep_indices_i \
                in zip(coords, cc_indices, sep_indices):
            entries = {}
            for cc in cc_indices_i:
                coord = coords_i[cc]
                ckey_score = np.zeros(2, dtype=np.float32)
                if coord is not None:
                    ckey_score[1] = 1.
                    bispan = coord.get_pair(cc, check=True)
                    assert bispan is not None
                else:
                    ckey_score[0] = 1.
                    bispan = None
                entry = {
                    'ckey_score': ckey_score,
                    'bispan': bispan,
                }
                entries[cc] = entry
            for sep in sep_indices_i:
                bispan = None
                for coord in coords_i.values():
                    if coord is not None and sep in coord.seps:
                        bispan = coord.get_pair(sep, check=True)
                        assert bispan is not None
                        break
                ckey_score = np.zeros(2, dtype=np.float32)
                ckey_score[int(bispan is not None)] = 1.
                entry = {
                    'ckey_score': ckey_score,
                    'bispan': bispan,
                }
                entries[sep] = entry
            self._data.append(entries)

    def get_entries(self, sentence_index):
        return self._data[sentence_index]

    def lookup_bispan_score(self, sentence_index, ckey, bispan):
        gold = self.get_entries(sentence_index)[ckey]['bispan']
        score = float(bispan == gold)
        return score
