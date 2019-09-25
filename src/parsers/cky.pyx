from cython.operator cimport dereference, postincrement


cdef class Grammar:

    def __init__(self):
        self.c = new _Grammar()

    def __dealloc__(self):
        del self.c

    def add_lexicon(self, str parent, unsigned item, int lex_type=0):
        self.c.add_lexicon(parent.encode(), item, lex_type)

    def add_unary(self, str parent, str child):
        self.c.add_unary(parent.encode(), child.encode())

    def add_binary(self, str parent, str left, str right):
        self.c.add_binary(parent.encode(), left.encode(), right.encode())


cdef class CkyParser:

    def __init__(self, Grammar grammar):
        self.c = new _CkyParser(dereference(grammar.c))
        self.grammar = grammar

    def __dealloc__(self):
        del self.c

    def start(self,
              const vector[unsigned] &words,
              const vector[pair[float, float]] &ckey_scores,
              unsigned n_best=1):
        self.c.start_parsing(words, ckey_scores, n_best)

    def resume(self):
        self.c.resume()

    @property
    def results(self):
        if not self.finished:
            return None
        cdef vector[pair[vector[_Coordination], float]] cpp_results
        cdef vector[pair[vector[_Coordination], float]].iterator it
        cdef vector[_Coordination].iterator coord_it
        cdef vector[_Coordination].iterator coord_it_end
        cpp_results = self.c.get_results()
        it = cpp_results.begin()
        results = []
        while it != cpp_results.end():
            coord_it = dereference(it).first.begin()
            coord_it_end = dereference(it).first.end()
            coords = []
            while coord_it != coord_it_end:
                cc = dereference(coord_it).cc
                conjuncts = [(conj.begin, conj.end)
                             for conj in dereference(coord_it).conjuncts]
                seps = dereference(coord_it).seps
                coords.append((cc, conjuncts, seps))
                postincrement(coord_it)
            results.append((coords, dereference(it).second))
            postincrement(it)
        return results

    @property
    def finished(self):
        return self.c.finished()

    @property
    def score_table(self):
        cdef cppmap[string, _ScoreTableValue] score_table \
            = self.c.get_score_table()
        cdef cppmap[string, _ScoreTableValue].iterator it \
            = score_table.begin()
        table = {}
        while it != score_table.end():
            key = dereference(it).first
            value = &dereference(it).second
            table[key] = (
                value.ckey,
                (value.left_conj.first, value.left_conj.second),
                (value.right_conj.first, value.right_conj.second),
                value.score,
            )
            postincrement(it)
        return table

    def update_score_table(self, const vector[_ScoreTableKV] &kvs):
        self.c.update_score_table(kvs)
