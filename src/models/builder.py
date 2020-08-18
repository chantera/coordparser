from models.common import CKeyClassifier, PairScorer
from models.feature import BaselineExtractor
from models.feature import FeatureExtractor1
from models.feature import FeatureExtractor2
from models.teranishi17 import Teranishi17
from models.teranishi19 import Teranishi19
from models.encoder import BiLSTMEncoder
from models.encoder import EmbeddingEncoder

from chainer_nn.links.nlp import BertBaseEmbedding
from chainer_nn.links.nlp import BertLargeEmbedding
from chainer_nn.links.nlp import ElmoEmbedding
import numpy as np


class EncoderBuilder(object):

    def __init__(self, loader, **kwargs):
        super().__setattr__('loader', loader)
        super().__setattr__('config', dict(
            inputs=('postag', 'char'),
            char_feature_size=50,
            char_pad_id=loader.char_pad_id,
            char_window_size=5,
            char_dropout=kwargs.get('dropout', 0.5),
            n_lstm_layers=2,
            lstm_hidden_size=512,
            embeddings_dropout=kwargs.get('dropout', 0.5),
            lstm_dropout=kwargs.get('dropout', 0.5),
            recurrent_dropout=0.0,
        ))
        self.config.update(kwargs)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            super().__setattr__(name, value)
        else:
            self.config[name] = value

    def __getattr__(self, name):
        if name not in self.config:
            raise AttributeError
        return self.config[name]

    def __setstate__(self, state):
        self.__dict__ = state

    def set(self, name, value):
        setattr(self, name, value)
        return self

    def build(self):
        loader = self.loader
        inputs = self.inputs

        contextualized_embeddings = None
        if sum(('elmo' in inputs,
                'bert-base' in inputs,
                'bert-large' in inputs)) > 1:
            raise ValueError(
                'at most 1 contextualized emebeddings can be chosen')
        elif 'elmo' in inputs:
            contextualized_embeddings = ElmoEmbedding(usage='weighted_sum')
        elif 'bert-base' in inputs:
            contextualized_embeddings \
                = BertBaseEmbedding(usage='second_to_last')
        elif 'bert-large' in inputs:
            contextualized_embeddings \
                = BertLargeEmbedding(usage='second_to_last')

        args = [
            loader.get_embeddings(
                'word', normalize=lambda W: W / np.std(W)
                if loader.use_pretrained_embed and np.std(W) > 0. else W),
            loader.get_embeddings('pos') if 'postag' in inputs else None,
            loader.get_embeddings('char') if 'char' in inputs else None,
            contextualized_embeddings,
            self.char_feature_size,
            self.char_pad_id,
            self.char_window_size,
            self.char_dropout,
            self.embeddings_dropout,
        ]

        if self.n_lstm_layers > 0:
            encoder = BiLSTMEncoder(*(args + [
                self.n_lstm_layers,
                self.lstm_hidden_size,
                self.lstm_dropout,
                self.recurrent_dropout,
            ]))
        else:
            encoder = EmbeddingEncoder(*args)
        return encoder

    def __repr__(self):
        return "{}(loader={}, config={})".format(
            self.__class__.__name__, repr(self.loader), repr(self.config))


class CoordSolverBuilder(EncoderBuilder):

    def __init__(self, loader, **kwargs):
        default_feature2 = 'extractor1' if kwargs.get('arch', 'Teranishi19') \
            in ('Teranishi19', 'teranishi19') else 'extractor2'
        super().__init__(loader, **kwargs)
        self.config.update(
            arch='Teranishi19',
            division='inner_outer',
            feature1='baseline',
            feature2=default_feature2,
            mlp_unit1=kwargs.get('lstm_hidden_size', 512) * 2,
            mlp_unit2=kwargs.get('lstm_hidden_size', 512) * 2,
            mlp_dropout=kwargs.get('dropout', 0.5),
        )
        self.config.update(kwargs)

    def build(self):
        encoder = super().build()

        extractors = []
        for feature in (self.feature1, self.feature2):
            if feature == 'baseline' or feature == BaselineExtractor:
                extractor = BaselineExtractor(encoder.out_size)
            elif feature == 'extractor1' or feature == FeatureExtractor1:
                extractor = FeatureExtractor1(encoder.out_size)
            elif feature == 'extractor2' or feature == FeatureExtractor2:
                extractor = FeatureExtractor2(encoder.out_size)
            else:
                raise ValueError("unknown feature: {}".format(feature))
            extractors.append(extractor)
        pair_scorer1 = PairScorer(
            extractors[0], self.mlp_unit1, self.mlp_dropout)
        pair_scorer2 = PairScorer(
            extractors[1], self.mlp_unit2, self.mlp_dropout)

        arch = self.arch
        if arch == "Teranishi19" or arch == "teranishi19":
            ckey_classifier = CKeyClassifier(encoder.out_size, 2)
            model = Teranishi19(
                encoder, ckey_classifier,
                pair_scorer1, pair_scorer2, self.division)
        elif arch == "Teranishi17" or arch == "teranishi17":
            ckey_classifier = CKeyClassifier(encoder.out_size, 1)
            model = Teranishi17(encoder, ckey_classifier, pair_scorer2)
        else:
            raise ValueError("unknown architecture: {}".format(arch))
        return model
