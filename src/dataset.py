import math
import os
import warnings

import numpy as np
from teras.dataset.dataset import Dataset
from teras.dataset.loader import CachedTextLoader
from teras.io import reader
from teras.preprocessing import text


_CHAR_PAD = "<PAD>"
CC_KEY = ["and", "or", "but", "nor", "and/or"]
CC_SEP = [",", ";", ":"]
OPEN_QUOTE = ("``", "`")
CLOSE_QUOTE = ("''", "'")


class DataLoader(CachedTextLoader):

    def __init__(self,
                 word_embed_size=100,
                 postag_embed_size=50,
                 char_embed_size=10,
                 word_embed_file=None,
                 word_preprocess=text.lower,
                 word_unknown="<UNK>",
                 filter_coord=False,
                 format='tree'):
        super().__init__(reader=None)
        self.init_reader(format)
        self.filter_coord = filter_coord
        self._updated = False
        self._postag_file = None
        self._cont_embed_file = None
        self._use_pretrained_embed = word_embed_file is not None

        word_vocab = text.EmbeddingVocab(
            file=word_embed_file,
            unknown=word_unknown, dim=word_embed_size,
            initializer=text.EmbeddingVocab.random_normal,
            serialize_embeddings=True)
        postag_vocab = text.EmbeddingVocab(
            unknown=word_unknown, dim=postag_embed_size,
            initializer=text.EmbeddingVocab.random_normal,
            serialize_embeddings=True)
        char_vocab = text.EmbeddingVocab(
            unknown=word_unknown, dim=char_embed_size,
            initializer=text.EmbeddingVocab.random_normal,
            serialize_embeddings=True)

        for word in CC_KEY + CC_SEP:
            word_vocab.add(word)
        self.char_pad_id = char_vocab.add(_CHAR_PAD)
        self.add_processor('word', word_vocab, preprocess=word_preprocess)
        self.add_processor('pos', postag_vocab, preprocess=False)
        self.add_processor('char', char_vocab, preprocess=False)

    def init_reader(self, format='default'):
        if format == 'tree':
            primary_reader = reader.TreeReader()
        elif format == 'genia':
            primary_reader = GeniaReader()
        else:
            primary_reader = reader.CsvReader(delimiter=' ')
        self._reader = reader.ZipReader([
            primary_reader,
            reader.CsvReader(delimiter=' '),
            reader.ContextualizedEmbeddingsReader(),
        ])

    @property
    def _primary_reader(self):
        return self._reader._readers[0]

    def map(self, item):
        words, postags, cont_embeds, coords \
            = _convert_item_to_attrs(item, type(self._primary_reader))
        (word_ids, postag_ids, char_ids,
         cc_indices, sep_indices, cont_embeds, indices, coords) \
            = _map(self, words, postags, cont_embeds, coords, swap_quote=True)
        self._updated = self.train

        for cc in cc_indices:
            if cc not in coords:
                coords[cc] = None
        # Check each separator belongs to only one coordination
        for sep in sep_indices:
            found_coord = False
            for coord in coords.values():
                if coord is None:
                    continue
                if sep in coord.seps:
                    assert not found_coord
                    found_coord = True

        return word_ids, postag_ids, char_ids, \
            cc_indices, sep_indices, cont_embeds, coords

    def filter(self, item):
        if self.filter_coord is False:
            return True

        if self.filter_coord is True or self.filter_coord == "any":
            if isinstance(self._primary_reader, reader.TreeReader):
                def _traverse(tree):
                    if len(tree) == 2 and isinstance(tree[1], str):  # Leaf
                        yield tree[1]
                    else:  # Node
                        for child in tree[1:]:
                            yield from _traverse(child)
                tree = item[0]
                word_iter = _traverse(tree[0] if len(tree) == 1 else tree)
            elif isinstance(self._primary_reader, GeniaReader):
                word_iter = (t['word'] for t in item[0][0])
            else:
                word_iter = (t.split('_')[0] for t in item[0])
            passed = _filter(word_iter, None, target='any')
        else:
            coords = _convert_item_to_attrs(
                item, type(self._primary_reader))[-1]
            passed = _filter(None, coords, target=self.filter_coord)

        return passed

    def load(self, file, train=False, size=None, bucketing=False,
             refresh_cache=False):
        self._updated = False
        files = [file, self._postag_file, self._cont_embed_file]
        dataset = super().load(
            files, train, size, bucketing,
            extra_ids=files[1:], refresh_cache=refresh_cache)
        if self._updated and self._cache_io is not None:
            self.update_cache()
        self._postag_file = self._cont_embed_file = None
        self._updated = False
        return dataset

    def load_with_external_resources(
            self, file, train=False, size=None, bucketing=False,
            refresh_cache=False,
            use_external_postags=False,
            use_contextualized_embed=False,
            postag_file_ext='.tag.ssv',
            contextualized_embed_file_ext='.hdf5',
            logger=None,
    ):
        if use_external_postags:
            postag_file = _find_file(file, postag_file_ext)
            if postag_file is not None and logger is not None:
                logger.info('load postags from {}'.format(postag_file))
            self.set_postag_file(postag_file)
        if use_contextualized_embed:
            cont_embed_file = _find_file(
                file, contextualized_embed_file_ext)
            if cont_embed_file is not None and logger is not None:
                logger.info('load contextualized embeddings from {}'
                            .format(cont_embed_file))
            self.set_contextualized_embed_file(cont_embed_file)
        return self.load(file, train, size, bucketing, refresh_cache)

    def set_postag_file(self, file):
        self._postag_file = file

    def set_contextualized_embed_file(self, file):
        self._cont_embed_file = file

    def use_pretrained_embed(self):
        return self._use_pretrained_embed

    def load_from_tagged_file(self, file, contextualized_embed_file=None):
        raise NotImplementedError


def _convert_item_to_attrs(item, reader_type):
    if issubclass(reader_type, reader.TreeReader):
        (tree, ext_postags, cont_embeds) = item
        words, postags, _spans, coords = _extract(tree)
    elif issubclass(reader_type, GeniaReader):
        ((sentence, coords), ext_postags, cont_embeds) = item
        words, postags = \
            zip(*[(token['word'], token['postag']) for token in sentence])
    else:
        sentence, ext_postags, cont_embeds = item
        words, postags = zip(*[token.split('_') for token in sentence])
        coords = {}

    if ext_postags is not None:
        assert len(words) == len(ext_postags)
        postags = ext_postags

    return words, postags, cont_embeds, coords


def _map(loader, words, postags, cont_embeds, coords=None,
         swap_quote=False):
    indices = np.arange(len(words), dtype=np.int32)
    swapped = False
    if swap_quote:
        for i, word in enumerate(words[:-1]):
            if word in CC_SEP and words[i + 1] in CLOSE_QUOTE:
                indices[i], indices[i + 1] = i + 1, i
                swapped = True

    if swapped:
        words = [words[idx] for idx in indices]
        postags = [postags[idx] for idx in indices]
    else:
        indices = None
    cc_indices = [i for i, word in enumerate(words) if word.lower() in CC_KEY]
    sep_indices = [i for i, word in enumerate(words) if word in CC_SEP]
    word_ids = loader.map_attr(
        'word', words, loader.train and not loader._use_pretrained_embed)
    char_ids = [loader.map_attr(
        'char', list(word), loader.train) for word in words]
    postag_ids = loader.map_attr('pos', postags, loader.train)
    cc_indices = np.array(cc_indices, np.int32)
    sep_indices = np.array(sep_indices, np.int32)

    if cont_embeds is not None:
        if swapped:
            # warnings.warn("contextualized embeddings are changed "
            #               "to swap quotation marks: sentence=`{}`"
            #               .format(' '.join(words)))
            cont_embeds = cont_embeds[:, indices]
        assert cont_embeds.shape[1] == len(words)
    if swapped and coords is not None:
        coords = preprocess(coords, indices)

    return word_ids, postag_ids, char_ids, \
        cc_indices, sep_indices, cont_embeds, indices, coords


def _filter(words, coords, target='any'):
    if target == 'any':
        return any(w.lower() in CC_KEY for w in words)

    n_conjuncts = [len(coord.conjuncts)
                   for coord in coords.values() if coord is not None]
    if len(n_conjuncts) == 0:
        passed = False
    elif target == 'simple':
        passed = len(n_conjuncts) == 1 and n_conjuncts[0] == 2
    elif target == 'not_simple':
        passed = (len(n_conjuncts) >= 2
                  or any(n > 2 for n in n_conjuncts))
    elif target == 'consecutive':
        passed = any(n > 2 for n in n_conjuncts)
    elif target == 'multiple':
        passed = len(n_conjuncts) >= 2
    else:
        raise ValueError("Invalid filter type: {}".format(target))
    return passed


def _find_file(path, ext):
    file = os.path.join(os.path.dirname(path),
                        os.path.basename(path).split('.')[0] + ext)
    return file if os.path.exists(file) else None


class GeniaReader(reader.Reader):
    COORD_FILE_EXT = '.coord'

    def __init__(self, file=None):
        super().__init__(file)
        if file is None:
            self.coord_file = None
            self.reset()

    def set_file(self, file):
        super().set_file(file)
        coord_file = _find_file(file, self.COORD_FILE_EXT)
        if not os.path.exists(coord_file):
            raise FileNotFoundError("coord file was not found: '{}'"
                                    .format(coord_file))
        self.coord_file = coord_file

    def __iter__(self):
        self._iterator = self._get_iterator()
        self._coord_iterator = self._get_coord_iterator()
        return self

    def __next__(self):
        try:
            sentence = self._iterator.__next__()
            coords = self._coord_iterator.__next__()
            if coords:
                words = [token['word'] for token in sentence]
                for coord in coords.values():
                    if coord is None or len(coord.conjuncts) <= 2:
                        continue
                    assert len(coord.seps) == 0
                    seps = _find_separators(words, coord.cc, coord.conjuncts)
                    coord.seps = tuple(seps)
            return (sentence, coords)
        except Exception as e:
            self.reset()
            raise e

    def read(self, file=None):
        if file is not None:
            self.set_file(file)
        items = [item for item in self]
        return items

    def read_next(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
            self._coord_iterator = self._get_coord_iterator()
        return self.__next__()

    def reset(self):
        super().reset()
        self._coord_iterator = None

    def _get_iterator(self):
        with open(self.file, mode='r', encoding='utf-8') as f:
            yield from _parse_genia(f)

    def _get_coord_iterator(self):
        with open(self.coord_file, mode='r', encoding='utf-8') as f:
            yield from _parse_genia_coord(f)

    def __getstate__(self):
        state = super().__getstate__()
        state['_coord_iterator'] = None
        return state


def _parse_genia(text):
    sentence_id = None
    tokens = []
    for line in [text] if isinstance(text, str) else text:
        line = line.strip()
        if not line:
            if len(tokens) > 0:
                yield tokens
                tokens = []
        elif line.startswith('#'):
            continue
        else:
            cols = line.split("\t")
            if cols[0] != sentence_id:
                if len(tokens) > 0:
                    yield tokens
                    tokens = []
                sentence_id = cols[0]
            token = {
                'id': int(cols[1]),
                'word': cols[2],
                'postag': cols[3],
            }
            tokens.append(token)
    if len(tokens) > 0:
        yield tokens


def _parse_genia_coord(text):
    sentence_id = None
    coord_id = None
    coord_key = None
    # coord_span = None
    coord_type = None
    segments = []
    coords = {}
    for line in [text] if isinstance(text, str) else text:
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        if cols[0].startswith('#'):
            assert cols[3] == cols[4]
            coord_key = int(cols[3]) - 1
            continue
        if (cols[0], cols[1]) != coord_id:
            if len(segments) > 0:
                assert len(segments) >= 2
                assert coord_key is not None
                coords[coord_key] = \
                    Coordination(coord_key, segments, label=coord_type)
                segments = []
            coord_id = (cols[0], cols[1])
            if cols[0] != sentence_id:
                if coords:
                    yield coords
                    coords = {}
                sentence_id = cols[0]
        if cols[2] == '*':
            # coord_span = (int(cols[3]) - 1, int(cols[4]) - 1)
            coord_type = cols[5].split('-')[0]
            continue
        segments.append((int(cols[3]) - 1, int(cols[4]) - 1))
    if segments:
        assert len(segments) >= 2
        assert coord_key is not None
        coords[coord_key] = Coordination(coord_key, segments, label=coord_type)
        yield coords


def _extract(tree):
    words = []
    postags = []
    spans = {}
    coords = {}

    def _traverse(tree, index):
        begin = index
        label = tree[0]
        if not label.startswith("-"):
            label = label.split("-")[0]
        if len(tree) == 2 and isinstance(tree[1], str):  # Leaf
            words.append(tree[1])
            postags.append(label)
        else:  # Node
            conjuncts = []
            cc = None
            for child in tree[1:]:
                child_label = child[0]
                assert child_label != "-NONE-"
                child_span = _traverse(child, index)
                if "COORD" in child_label:
                    conjuncts.append(child_span)
                elif child_label == "CC" \
                    or (child_label.startswith("CC-")
                        and child_label != "CC-SHARED"):
                    assert isinstance(child[1], str)
                    cc = child_span[0]
                index = child_span[1] + 1
            if cc is not None and len(conjuncts) >= 2:
                seps = _find_separators(words, cc, conjuncts)
                coords[cc] = Coordination(cc, conjuncts, seps, label)
            index -= 1
        span = (begin, index)
        if span not in spans:
            spans[span] = [label]
        else:
            spans[span].append(label)
        return span

    _traverse(tree[0] if len(tree) == 1 else tree, index=0)
    return words, postags, spans, coords


def _find_separators(words, cc, conjuncts):
    seps = []
    if len(conjuncts) > 2:
        for i in range(1, len(conjuncts) - 1):
            sep = _find_separator(words,
                                  conjuncts[i - 1][1] + 1,
                                  conjuncts[i][0],
                                  search_len=2)
            if sep is None:
                warnings.warn(
                    "Could not find separator: "
                    "left conjunct={}, right conjunct={}, "
                    "range: {}".format(
                        conjuncts[i - 1], conjuncts[i],
                        words[conjuncts[i - 1][0]:
                              conjuncts[i][1] + 1]))
                continue
            elif sep == cc:
                continue
            seps.append(sep)
    return seps


def _find_separator(words, search_from, search_to, search_len=2):
    """
    NOTE: `search_from` is inclusive but `search_to` is not inclusive
    """
    assert search_len > 1
    diff = search_to - search_from
    if diff < 1:
        return None
    half = math.ceil(diff / 2)
    if half < search_len:
        search_len = half
    for i in range(search_len):
        if words[search_to - 1 - i].lower() in CC_KEY:
            return search_to - 1 - i
        elif words[search_from + i].lower() in CC_KEY:
            return search_from + i
    for i in range(search_len):
        if words[search_to - 1 - i] in CC_SEP:
            return search_to - 1 - i
        elif words[search_from + i] in CC_SEP:
            return search_from + i
    return None


class Coordination(object):
    __slots__ = ('cc', 'conjuncts', 'seps', 'label')

    def __init__(self, cc, conjuncts, seps=None, label=None,
                 suppress_warning=False):
        assert isinstance(conjuncts, (list, tuple)) and len(conjuncts) >= 2
        assert all(isinstance(conj, tuple) for conj in conjuncts)
        conjuncts = sorted(conjuncts, key=lambda span: span[0])
        # NOTE(chantera): The form 'A and B, C' is considered to be coordination.  # NOQA
        # assert cc > conjuncts[-2][1] and cc < conjuncts[-1][0]
        assert cc > conjuncts[0][1] and cc < conjuncts[-1][0]
        if seps is not None:
            if len(seps) == len(conjuncts) - 2:
                for i, sep in enumerate(seps):
                    assert conjuncts[i][1] < sep and conjuncts[i + 1][0] > sep
            elif not suppress_warning:
                warnings.warn(
                    "Coordination does not contain enough separators. "
                    "It may be a wrong coordination: "
                    "cc={}, conjuncts={}, separators={}"
                    .format(cc, conjuncts, seps))
        else:
            seps = []
        self.cc = cc
        self.conjuncts = tuple(conjuncts)
        self.seps = tuple(seps)
        self.label = label

    def get_pair(self, index, check=False):
        pair = None
        for i in range(1, len(self.conjuncts)):
            if self.conjuncts[i][0] > index:
                pair = (self.conjuncts[i - 1], self.conjuncts[i])
                assert pair[0][1] < index and pair[1][0] > index
                break
        if check and pair is None:
            raise LookupError(
                "Could not find any pair for index={}".format(index))
        return pair

    def __repr__(self):
        return "Coordination(cc={}, conjuncts={}, seps={}, label={})".format(
            self.cc, self.conjuncts, self.seps, self.label)

    def __eq__(self, other):
        if not isinstance(other, Coordination):
            return False
        return self.cc == other.cc \
            and len(self.conjuncts) == len(other.conjuncts) \
            and all(conj1 == conj2 for conj1, conj2
                    in zip(self.conjuncts, other.conjuncts))


def preprocess(coords, indices):
    new_coords = {}
    for cc, coord in coords.items():
        cc = indices[cc]
        if coord is not None:
            seps = [indices[s] for s in coord.seps]
            conjuncts = []
            n = len(coord.conjuncts)
            for i, (b, e) in enumerate(coord.conjuncts):
                if indices[b] == b - 1 and indices[b - 1] == b:
                    # Special case for annotation error
                    new_b = b + 1
                else:
                    new_b = indices[b]
                if indices[e] == e - 1 and indices[e - 1] == e:
                    # Case: ``A,'' ``B,'' and ``C,'' ; (0, 3), (4, 7), (9, 12)
                    #    => ``A'', ``B'', and ``C'', ; (0, 2), (4, 6), (9, 11)
                    new_e = indices[e]
                    sep = e
                    if i < n - 2 and sep + 1 != cc and sep not in seps:
                        seps.append(sep)
                elif indices[e] == e + 1 and indices[e + 1] == e:
                    # Case: ``A,'' ``B,'' and ``C,'' ; (1, 2), (5, 6), (10, 11)
                    #    => ``A'', ``B'', and ``C'', ; (1, 1), (5, 5), (10, 10)
                    new_e = e - 1
                    sep = e + 1
                    if i < n - 2 and sep + 1 != cc and sep not in seps:
                        seps.append(sep)
                else:
                    new_e = indices[e]
                conjuncts.append((new_b, new_e))
            coord = Coordination(cc, conjuncts, seps, coord.label)
        new_coords[cc] = coord
    return new_coords


def postprocess(coords, indices):
    new_coords = {}
    for cc, coord in coords.items():
        cc = indices[cc]
        if coord is not None:
            seps = [indices[s] for s in coord.seps]
            conjuncts = []
            n = len(coord.conjuncts)
            for i, (b, e) in enumerate(coord.conjuncts):
                new_b = indices[b]
                if indices[e] == e + 1 and indices[e + 1] == e:
                    # Case: ``A'', ``B'', and ``C'', ; (0, 2), (4, 6), (9, 11)
                    #    => ``A,'' ``B,'' and ``C,'' ; (0, 3), (4, 7), (9, 12)
                    new_e = indices[e]
                    sep = e
                    if i < n - 1 and sep in seps:
                        seps.remove(sep)
                elif e < indices.size - 1 \
                        and indices[e + 1] == e + 2 \
                        and indices[e + 2] == e + 1:
                    # Case: ``A'', ``B'', and ``C'', ; (1, 1), (5, 5), (10, 10)
                    #    => ``A,'' ``B,'' and ``C,'' ; (1, 1), (5, 5), (10, 10)
                    new_e = e  # always exclude the trailing comma
                    sep = e + 1
                    if i < n - 1 and sep in seps:
                        seps.remove(sep)
                else:
                    new_e = indices[e]
                conjuncts.append((new_b, new_e))
            coord = Coordination(cc, conjuncts, seps, coord.label,
                                 suppress_warning=True)
        new_coords[cc] = coord
    return new_coords
