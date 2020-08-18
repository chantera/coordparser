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

    def __init__(self, in_size, use_sim=True, use_repl=True):
        super().__init__()
        self.in_size = in_size
        self.use_sim = use_sim
        self.use_repl = use_repl

    def forward(self, hs_flatten, pairs, ckeys, lengths):
        features = []
        if self.use_sim:
            features.extend(
                self._feature_sim(hs_flatten, pairs, ckeys, lengths))
        if self.use_repl:
            features.extend(
                self._feature_repl(hs_flatten, pairs, ckeys, lengths))
        if not (self.use_sim or self.use_repl):
            features.extend(
                self._forward_spans(hs_flatten, pairs, ckeys, lengths))
        fs = F.hstack(features)
        return fs

    @staticmethod
    def _feature_sim(hs_flatten, pairs, ckeys, lengths):
        left_spans, right_spans = FeatureExtractor2._forward_spans(
            hs_flatten, pairs, ckeys, lengths)
        sim1 = F.absolute(left_spans - right_spans)
        sim2 = left_spans * right_spans
        return sim1, sim2

    @staticmethod
    def _feature_repl(hs_flatten, pairs, ckeys, lengths):
        xp = chainer.cuda.get_array_module(hs_flatten)
        begins, ends = pairs.T
        begins_ = xp.asarray(begins)
        ends_ = xp.asarray(ends)
        ckeys_ = xp.asarray(ckeys)

        h_b = F.embed_id(begins_, hs_flatten)
        h_b_pre = F.embed_id(begins_ - 1, hs_flatten, ignore_label=-1)
        out_of_span = np.insert(lengths[:-1].cumsum(), 0, 0) - 1
        is_out_of_span = np.isin(begins - 1, out_of_span)
        h_b_pre = F.where(xp.asarray(is_out_of_span)[:, None],
                          xp.zeros_like(h_b_pre.data), h_b_pre)
        h_e = F.embed_id(ends_, hs_flatten)
        h_e_post = F.embed_id(ends_ + 1, hs_flatten, hs_flatten.shape[0])
        out_of_span = lengths.cumsum()
        is_out_of_span = np.isin(ends + 1, out_of_span)
        h_e_post = F.where(xp.asarray(is_out_of_span)[:, None],
                           xp.zeros_like(h_e_post.data), h_e_post)
        h_k_pre = F.embed_id(ckeys_ - 1, hs_flatten)
        h_k_post = F.embed_id(ckeys_ + 1, hs_flatten)

        repl1 = F.absolute(h_b_pre * (h_b - h_k_post))
        repl2 = F.absolute(h_e_post * (h_e - h_k_pre))
        return repl1, repl2

    @staticmethod
    def _forward_spans(hs_flatten, pairs, ckeys, lengths):
        begins, ends = pairs.T

        @functools.lru_cache(maxsize=None)
        def _get_span_v(i, j):
            return F.average(hs_flatten[i:j + 1], axis=0)

        left_spans = F.vstack(
            [_get_span_v(begin, ckey_pre) for begin, ckey_pre
             in zip(begins, ckeys - 1)])
        right_spans = F.vstack(
            [_get_span_v(ckey_post, end) for ckey_post, end
             in zip(ckeys + 1, ends)])

        return left_spans, right_spans

    @property
    def out_size(self):
        n_features = 2 * max(int(self.use_sim) + int(self.use_repl), 1)
        return self.in_size * n_features
