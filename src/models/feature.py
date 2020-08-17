import functools

import chainer
import chainer.functions as F
import numpy as np

from models.common import FeatureExtractor


class BaselineExtractor(FeatureExtractor):

    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size

    def forward(self, hs_flatten, pairs, ckeys, lengths):
        xp = chainer.cuda.get_array_module(hs_flatten)
        fs = F.embed_id(xp.asarray(pairs.reshape(-1)), hs_flatten) \
            .reshape(pairs.shape[0], self.out_size)
        return fs

    @property
    def out_size(self):
        return self.in_size * 2


class FeatureExtractor1(FeatureExtractor):

    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size

    def forward(self, hs_flatten, pairs, ckeys, lengths):
        xp = chainer.cuda.get_array_module(hs_flatten)
        p1, p2 = xp.asarray(pairs.T)
        ckeys = xp.asarray(ckeys)
        h_p1 = F.embed_id(p1, hs_flatten)
        h_p2 = F.embed_id(p2, hs_flatten)
        h_cnext = F.embed_id(ckeys + 1, hs_flatten)
        h_cprev = F.embed_id(ckeys - 1, hs_flatten)
        fs = F.concat((h_p1 - h_cnext, h_p2 - h_cprev), axis=1)
        return fs

    @property
    def out_size(self):
        return self.in_size * 2


class FeatureExtractor2(FeatureExtractor):

    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size

    def forward(self, hs_flatten, pairs, ckeys, lengths):
        xp = chainer.cuda.get_array_module(hs_flatten)
        begins, ends = xp.asarray(pairs.T)
        ckeys, lengths = xp.asarray(ckeys), xp.asarray(lengths)
        begins_pre = begins - 1
        ends_post = ends + 1
        ckeys_pre = ckeys - 1
        ckeys_post = ckeys + 1
        to_cpu = chainer.cuda.to_cpu

        @functools.lru_cache(maxsize=None)
        def _get_span_v(i, j):
            return F.average(hs_flatten[i:j + 1], axis=0)

        left_spans = F.vstack([_get_span_v(begin, ckey_pre)
                               for begin, ckey_pre
                               in zip(to_cpu(begins), to_cpu(ckeys_pre))])
        right_spans = F.vstack([_get_span_v(ckey_post, end)
                                for ckey_post, end
                                in zip(to_cpu(ckeys_post), to_cpu(ends))])
        h_b = F.embed_id(begins, hs_flatten)
        h_b_pre = F.embed_id(begins_pre, hs_flatten, ignore_label=-1)
        out_of_span = np.insert(to_cpu(lengths[:-1].cumsum()), 0, 0) - 1
        is_in = np.isin(to_cpu(begins_pre), out_of_span)
        h_b_pre = F.where(xp.asarray(is_in)[:, None],
                          xp.zeros_like(h_b_pre.data), h_b_pre)
        h_e = F.embed_id(ends, hs_flatten)
        h_e_post = F.embed_id(ends_post, hs_flatten, hs_flatten.shape[0])
        out_of_span = lengths.cumsum()
        is_in = np.isin(to_cpu(ends_post), to_cpu(out_of_span))
        h_e_post = F.where(xp.asarray(is_in)[:, None],
                           xp.zeros_like(h_e_post.data), h_e_post)
        h_k_pre = F.embed_id(ckeys_pre, hs_flatten)
        h_k_post = F.embed_id(ckeys_post, hs_flatten)

        sim1 = F.absolute(left_spans - right_spans)
        sim2 = left_spans * right_spans
        repl1 = F.absolute(h_b_pre * (h_b - h_k_post))
        repl2 = F.absolute(h_e_post * (h_e - h_k_pre))
        fs = F.hstack((sim1, sim2, repl1, repl2))
        return fs

    @property
    def out_size(self):
        return self.in_size * 4
