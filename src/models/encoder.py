from itertools import chain

import chainer
import chainer.functions as F
import chainer.links as L
import chainer_nn.functions as nn_F
import chainer_nn.links as nn_L
import numpy as np


class Encoder(chainer.Chain):

    def __init__(self,
                 word_embeddings,
                 postag_embeddings=None,
                 char_embeddings=None,
                 contextualized_embeddings=None,
                 char_feature_size=50,
                 char_pad_id=1,
                 char_window_size=5,
                 char_dropout=0.0,
                 embeddings_dropout=0.0):
        super().__init__()
        with self.init_scope():
            embeddings = {'word_embed': word_embeddings,
                          'postag_embed': postag_embeddings}
            in_size = 0
            for name, weights in embeddings.items():
                if weights is not None:
                    assert weights.ndim == 2
                    s = weights.shape
                    self.__setattr__(name, L.EmbedID(s[0], s[1], weights))
                    in_size += s[1]
                else:
                    self.__setattr__(name, None)
            if char_embeddings is not None:
                self.char_cnn = nn_L.CharCNN(
                    char_embeddings, char_pad_id, char_feature_size,
                    char_window_size, char_dropout)
                in_size += char_feature_size
            else:
                self.char_cnn = None
            if contextualized_embeddings is not None:
                self.cont_embed = contextualized_embeddings
                in_size += contextualized_embeddings.out_size
            else:
                self.cont_embed = None
        self.embeddings_dropout = embeddings_dropout
        self._in_size = in_size

    def forward(self, words, postags=None, chars=None, cont_embeds=None):
        lengths = np.array([seq.size for seq in words], np.int32)
        xs = [self._forward_embed(embed, x) for embed, x
              in ((self.word_embed, words),
                  (self.postag_embed, postags))
              if embed is not None]
        if self.char_cnn is not None:
            xs.append(self.char_cnn(list(chain.from_iterable(chars))))
        if self.cont_embed is not None:
            xp = chainer.cuda.get_array_module(cont_embeds[0])
            xs.append(self.cont_embed.forward_one(
                xp.concatenate(cont_embeds, axis=1)))
        xs = F.concat(xs) if len(xs) > 1 else xs[0]
        xs = nn_F.dropout(xs, self.embeddings_dropout)
        xs = F.split_axis(xs, lengths[:-1].cumsum(), axis=0)
        hs = self._encode_sequence(xs, lengths)
        return hs, lengths

    def _encode_sequence(self, xs, lengths):
        raise NotImplementedError

    @staticmethod
    def _forward_embed(embed, x):
        xp = chainer.cuda.get_array_module(x[0])
        return embed(embed.xp.asarray(xp.concatenate(x, axis=0)))

    @property
    def out_size(self):
        raise NotImplementedError


class EmbeddingEncoder(Encoder):

    def _encode_sequence(self, xs, lengths):
        return xs

    @property
    def out_size(self):
        return self._in_size


class BiLSTMEncoder(Encoder):

    def __init__(self,
                 word_embeddings,
                 postag_embeddings=None,
                 char_embeddings=None,
                 contextualized_embeddings=None,
                 char_feature_size=50,
                 char_pad_id=1,
                 char_window_size=5,
                 char_dropout=0.0,
                 embeddings_dropout=0.0,
                 n_lstm_layers=2,
                 lstm_hidden_size=200,
                 lstm_dropout=0.0,
                 recurrent_dropout=0.0):
        super().__init__(word_embeddings, postag_embeddings,
                         char_embeddings, contextualized_embeddings,
                         char_feature_size, char_pad_id, char_window_size,
                         char_dropout, embeddings_dropout)
        with self.init_scope():
            self.bilstm = nn_L.NStepBiLSTM(
                n_lstm_layers, self._in_size, lstm_hidden_size,
                lstm_dropout, recurrent_dropout)
        self.lstm_dropout = lstm_dropout
        self._hidden_size = lstm_hidden_size

    def _encode_sequence(self, xs, lengths):
        hs = self.bilstm(hx=None, cx=None, xs=xs)[-1]
        # NOTE(chantera): Disable to reproduce [Teranishi et al., 2019].
        # hs = nn_F.dropout(F.pad_sequence(hs), self.lstm_dropout)
        return hs

    @property
    def out_size(self):
        return self._hidden_size * 2
