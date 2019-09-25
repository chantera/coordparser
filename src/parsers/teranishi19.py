from collections import defaultdict

import numpy as np

from dataset import Coordination
from parsers import cky
from parsers.common import Parser


class Grammar(cky.Grammar):
    CFG = [
        ("COORD", ("CJT", "CC", "CJT")),
        ("COORD", ("CJT", "CC-SEP", "COORD")),
        ("CJT", ("COORD",)),
        ("CJT", ("N",)),
        ("S", ("COORD",)),
        ("S", ("N",)),
        ("N", ("COORD", "N",)),
        ("N", ("W", "COORD")),
        ("N", ("W", "N")),
        ("N", ("W",)),
        ("CC", "and"),
        ("CC", "or"),
        ("CC", "but"),
        ("CC", "nor"),
        ("CC", "and\/or"),
        ("CC-SEP", ","),
        ("CC-SEP", ";"),
        ("CC-SEP", ":"),
        ("W", "<ANY>"),
    ]
    CFG_COORD_1 = [
        ("COORD", ("CJT", "W", "CC", "CJT")),
        ("COORD", ("CJT", "CC", "W", "CJT")),
        ("COORD", ("CJT", "W", "CC", "W", "CJT")),
    ]
    CFG_COORD_2 = [
        ("COORD", ("CJT", "N", "CC", "CJT")),
        ("COORD", ("CJT", "CC", "N", "CJT")),
        ("COORD", ("CJT", "N", "CC", "N", "CJT")),
    ]

    def __init__(self, word_vocab=None, cfg=None):
        super().__init__()
        if word_vocab is None:
            word_vocab = defaultdict(lambda: len(word_vocab))
        if cfg is None:
            cfg = self.CFG_COORD_2 + self.CFG
        for rule in reversed(cfg):
            self.add_rule(rule, word_vocab)

    def add_rule(self, rule, word_vocab):
        parent, non_terminals = rule
        if isinstance(non_terminals, str):  # lexicon
            if non_terminals == "<ANY>":
                lex_type = 1
            elif parent == "CC":
                lex_type = 2
            elif parent == "CC-SEP":
                lex_type = 3
            else:
                lex_type = 0
            self.add_lexicon(parent, word_vocab[non_terminals], lex_type)
            return
        assert isinstance(non_terminals, tuple)
        n_non_terminals = len(non_terminals)
        if n_non_terminals == 1:  # unary
            self.add_unary(parent, non_terminals[0])
        elif n_non_terminals == 2:  # binary
            self.add_binary(parent, non_terminals[0], non_terminals[1])
        else:  # n-ary (n >= 3)
            assert n_non_terminals >= 3
            for parent, left, right in _binarize(parent, non_terminals):
                self.add_binary(parent, left, right)


def _binarize(parent, non_terminals):
    rules = []
    # NOTE(chantera): This label formation may cause conflict between unscored constituents.  # NOQA
    """
    # right binarization
    non_terminals = list(reversed(non_terminals))
    while non_terminals:
        left = non_terminals.pop()
        right = "{}\\{}".format(parent, left) \
            if len(non_terminals) > 1 else non_terminals.pop()
        rules.append((parent, left, right))
        parent = right
    """
    # right binarization
    non_terminals = list(non_terminals)
    non_terminals.insert(0, parent)
    right = non_terminals.pop()
    while non_terminals:
        left = non_terminals.pop()
        parent = "({}:{})".format(left, right) \
            if len(non_terminals) > 1 else non_terminals.pop()
        rules.append((parent, left, right))
        right = parent
    rules.reverse()
    return rules


class CkyParser(Parser):

    def __init__(self, model, grammar):
        self.parser_impl = cky.CkyParser(grammar)
        self.model = model
        self._batch_index = -1

    def parse(self, words, postags, chars, cc_indices, sep_indices,
              cont_embeds=None, n_best=1, use_cache=False):
        coords_batch = []
        if not use_cache or not self.model.has_cache():
            self.model.forward(words, postags, chars,
                               cc_indices, sep_indices, cont_embeds)
        for batch_index in range(len(words)):
            if len(cc_indices[batch_index]) == 0:
                coords_batch.append([])
                continue
            self._batch_index = batch_index
            results = self._parse_each(words[batch_index],
                                       cc_indices[batch_index],
                                       sep_indices[batch_index],
                                       n_best)
            coords_batch.append(results)
        return coords_batch

    def _parse_each(self, words, cc_indices, sep_indices, n_best):
        self.parser_impl.start(words, self._get_ckey_scores(words), n_best)
        while not self.parser_impl.finished:
            self._assign_pair_scores(self.parser_impl.score_table)
            self.parser_impl.resume()
        raw_results = self.parser_impl.results
        assert raw_results is not None
        results = []
        for (coords, score) in raw_results:
            coord_dict = {}
            for (cc, conjuncts, seps) in coords:
                coord_dict[cc] = Coordination(cc, conjuncts, seps)
            for cc in cc_indices:
                if cc not in coord_dict:
                    coord_dict[cc] = None
            results.append((coord_dict, score))
        return results

    def _get_ckey_scores(self, words):
        scores = np.zeros((len(words), 2), dtype=np.float32)
        entries = self.model.score_table.get_entries(self._batch_index)
        for ckey, values in entries.items():
            scores[ckey] = values['ckey_score']
        return scores

    def _assign_pair_scores(self, table):
        lookup = self.model.score_table.lookup_bispan_score
        index = self._batch_index
        new_scores = [(key, lookup(index, ckey, (left_conj, right_conj)))
                      for key, (ckey, left_conj, right_conj, _score)
                      in table.items()]
        self.parser_impl.update_score_table(new_scores)
