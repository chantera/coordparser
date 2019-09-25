from models.teranishi17 import Teranishi17
from models.teranishi19 import Teranishi19
from parsers.teranishi17 import Teranishi17Parser
from parsers.teranishi19 import Grammar
from parsers.teranishi19 import CkyParser


def build_parser(loader, model):
    word_vocab = loader.get_processor('word').vocab
    if isinstance(model, Teranishi17):
        parser = Teranishi17Parser(
            model, comma_id=word_vocab[','])
    elif isinstance(model, Teranishi19):
        grammar = Grammar(word_vocab)
        parser = CkyParser(model, grammar)
    else:
        raise ValueError("Unsupported model: {}"
                         .format(model.__class__.__name__))
    return parser
