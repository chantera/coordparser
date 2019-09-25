from libcpp cimport bool
from libcpp.map cimport map as cppmap
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cky.h" nogil:
    cdef cppclass _Coordination "coord::Coordination":
        unsigned cc;
        vector[_Conjunct] conjuncts;
        vector[long unsigned] seps;


cdef extern from "cky.h" nogil:
    cdef cppclass _Conjunct "coord::Coordination::Conjunct":
        unsigned begin;
        unsigned end;


cdef extern from "cky.h" nogil:
    cdef cppclass _Grammar "coord::Grammar":
        _Grammar()
        void add_lexicon(const string &parent, unsigned item, int lex_type)
        void add_unary(const string &parent, const string &child)
        void add_binary(const string &parent, const string &left, const string &right)


cdef class Grammar:
    cdef _Grammar *c


ctypedef pair[string, float] _ScoreTableKV


cdef extern from "cky.h" nogil:
    cdef cppclass _ScoreTableValue "coord::ScoreTableValue":
        const unsigned ckey;
        const pair[unsigned, unsigned] left_conj;
        const pair[unsigned, unsigned] right_conj;
        float score;


cdef extern from "cky.h" nogil:
    cdef cppclass _CkyParser "coord::CkyParser":
        _CkyParser(const _Grammar &grammar) except +
        _CkyParser(const _Grammar &grammar,
                   unsigned tag_complete,
                   unsigned tag_coord,
                   unsigned tag_conj,
                   unsigned tag_cc,
                   unsigned tag_sep) except +
        void start_parsing(const vector[unsigned] &words,
                           const vector[pair[float, float]] &ckey_scores,
                           unsigned n_best) except +
        vector[pair[vector[_Coordination], float]] get_results() except +
        void resume() except +
        bool finished() const
        const cppmap[string, _ScoreTableValue] &get_score_table() const
        void update_score_table(const vector[_ScoreTableKV] &kvs) except +


cdef class CkyParser:
    cdef _CkyParser *c
    cdef Grammar grammar
