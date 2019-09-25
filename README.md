# Coordparser

This is an implementation of "[Decomposed Local Models for Coordinate Structure Parsing](https://www.aclweb.org/anthology/N19-1343)".

## Installation

```
pip install -r requirements.txt
make
```

## Training

### Usage

```
usage: main.py train [-h] [--batchsize NUM] [--cachedir DIR] [--devfile FILE]
                     [--device ID] [--embedfile FILE] [--epoch NUM]
                     [--format {tree,genia}] [--gradclip VALUE]
                     [--inputs [{char,postag,elmo,bert-base,bert-large} [{char,postag,elmo,bert-base,bert-large} ...]]]
                     [--l2 VALUE] [--limit NUM] [--lr VALUE]
                     [--model KEY=VALUE] [--refresh] [--savedir DIR]
                     [--seed VALUE] --trainfile FILE

optional arguments:
  -h, --help            show this help message and exit
  --batchsize NUM       Number of examples in each mini-batch (default: 20)
  --cachedir DIR        Cache directory (default: cache)
  --devfile FILE        Development data file (default: None)
  --device ID           Device ID (negative value indicates CPU) (default: -1)
  --embedfile FILE      Pretrained word embedding file (default: None)
  --epoch NUM           Number of sweeps over the dataset to train (default:
                        20)
  --format {tree,genia}
                        Training/Development data format (default: tree)
  --gradclip VALUE      L2 norm threshold of gradient norm (default: 5.0)
  --inputs [{char,postag,elmo,bert-base,bert-large} [{char,postag,elmo,bert-base,bert-large} ...]]
                        Additional inputs for the encoder (default: ('char',
                        'postag'))
  --l2 VALUE            Strength of L2 regularization (default: 0.0)
  --limit NUM           Limit of the number of training samples (default: -1)
  --lr VALUE            Learning Rate (default: 0.001)
  --model KEY=VALUE     Model configuration (default: None)
  --refresh, -r         Refresh cache (default: False)
  --savedir DIR         Directory to save the model (default: None)
  --seed VALUE          Random seed (default: None)
  --trainfile FILE      Training data file (default: None)
```

### Training/Development data format

#### Tree format

In the tree format, a text file looks like following.

```
(S (NP-SBJ (NP (JJ Influential)(NNS members))(PP (IN of)(NP (DT the)(NNP House)(NP-CCP (NNP-COORD Ways)(CC-CC and)(NNP-COORD Means))(NNP Committee))))(VP (VBD introduced)(NP (NP (NN legislation))(SBAR (WHNP-1 (WDT that))(S (VP (MD would)(VP (VB restrict)(SBAR (WHADVP-2 (WRB how))(S (NP-SBJ (DT the)(JJ new)(NML (NN savings-and-loan)(NN bailout))(NN agency))(VP (MD can)(VP (VB raise)(NP (NN capital))))))(, ,)(S-ADV (VP (VBG creating)(NP (NP (DT another)(JJ potential)(NN obstacle))(PP (TO to)(NP (NP (NP (NML (DT the)(NN government))(POS 's))(NN sale))(PP (IN of)(NP (JJ sick)(NNS thrifts))))))))))))))(. .))
...
```

It is not necessary to represent one tree in one line.
A tree, however, must not contain nodes labeled as ``` `` ``` or `''`.
To exclude such nodes, use `data/clean.py`.

Coordinate structures consist of `CC` and `COORD` nodes.
For further information of the annotation scheme, please refer to "[Coordination Annotation Extension in the Penn Tree Bank](https://www.aclweb.org/anthology/P16-1079)".

#### GENIA format

The GENIA format is a special representation used in the original code of "[Coordinate Structure Analysis with Global Structural Constraints and Alignment-Based Local Features](https://www.aclweb.org/anthology/P09-1109)".
In the GENIA format, a sentence and its coordination are annotated in separated files.
Each line in a sentence file has sentence ID, word index, word and POS tag fields separated by tabs.

```
4321	1	Positive	JJ
4321	2	and	CC
4321	3	negative	JJ
4321	4	regulation	NN
4321	5	of	IN
4321	6	immunoglobulin	NN
4321	7	gene	NN
4321	8	expression	NN
4321	9	by	IN
4321	10	a	DT
4321	11	novel	JJ
4321	12	B-cell-specific	JJ
4321	13	enhancer	NN
4321	14	element	NN
4321	15	.	.

...
```

In the corresponding coordination file, coordinate structures are represented as following.

```
4321	1	*	1	3	ADJP-COOD
4321	1	1	1	1	ADJP-COOD
# 4321	1	2	2	2	ADJP-COOD
4321	1	3	3	3	ADJP-COOD

...
5432	1	*	14	36	VP-COOD
5432	1	1	14	19	VP-COOD
# 5432	1	2	20	20	VP-COOD
5432	1	3	21	25	VP-COOD
# 5432	1	4	26	26	VP-COOD
# 5432	1	5	27	27	VP-COOD
5432	1	6	28	36	VP-COOD

...
```

Lines have the following fields separated by tabs.
1. Sentence id
2. Coordination number
3. Conjunct number (`*` indicates the coordinate structure itself)
4. Beginning of the span
5. End of the span
6. Type of the span

A line beginning with `#` is not a comment, but regards its span as a separator between conjuncts.

To use the GENIA format files, give a sentence file to `--trainfile/--devfile`, put the corresponding coordination file in the same directory of the sentence file with the `.coord` file extension, and specify `--format=genia`.

### How to use external part-of-speech tags

To replace POS tags with those from an external file, put the file in the same directory of `--trainfile/--devfile` with the same basename combined with `.tag.ssv` file extension.
A sequence of POS tags for a sentence is placed in a line using single white spaces as a delimiter.
POS tag files are automatically loaded when available.

### How to use contextualized embeddings

Training with contextualized embedding is enabled by including `elmo`, `bert-base` or `bert-large` in `--inputs`.
If so, put the contextualized embeddings files in the same directory of `--trainfile/--devfile` with the same basename combined with `.elmo.hdf5`, `.bert-base.hdf5` or `.bert-large.hdf5` file extension.
Do not forget to include other inputs like `char` and `postag` in `--inputs` if needed.

Contextualized embeddings files in the HDF5 format are easily obtained by [chantera/chainer_contextualized_embeddings](https://github.com/chantera/chainer_contextualized_embeddings).
`data/extract.py` helps you extract raw sentences from a tree file.

## Evaluation

### Usage

```
usage: main.py test [-h] [--device ID]
                    [--filter {any,simple,not_simple,consecutive,multiple}]
                    [--limit NUM] --modelfile FILE --testfile FILE

optional arguments:
  -h, --help            show this help message and exit
  --device ID           Device ID (negative value indicates CPU) (default: -1)
  --filter {any,simple,not_simple,consecutive,multiple}
                        Filter type for sentence (default: any)
  --limit NUM           Limit of the number of training samples (default: -1)
  --modelfile FILE      Trained model file (default: None)
  --testfile FILE       Test data file (default: None)
```

## Parsing

### Usage

```
usage: main.py parse [-h] [--cembfile FILE] [--device ID] --input FILE
                     --modelfile FILE [--nbest NUM]

optional arguments:
  -h, --help        show this help message and exit
  --cembfile FILE   Contextualized embeddings file (default: None)
  --device ID       Device ID (negative value indicates CPU) (default: -1)
  --input FILE      Input text file to parse (default: None)
  --modelfile FILE  Trained model file (default: None)
  --nbest NUM       Number of candidates to output (default: 1)
```

### Data format

In a target file, each sentence is represented as a sequence of `WORD_POSTAG` tokens.

```
Influential_JJ members_NNS of_IN the_DT House_NNP Ways_NNP and_CC Means_NNP Committee_NNP introduced_VBD legislation_NN that_WDT would_MD restrict_VB how_WRB the_DT new_JJ savings-and-loan_NN bailout_NN agency_NN can_MD raise_VB capital_NN ,_, creating_VBG another_DT potential_JJ obstacle_NN to_TO the_DT government_NN 's_POS sale_NN of_IN sick_JJ thrifts_NNS ._.
...
```

Unlike training/development files for training, a sentence can contain quotation marks.

## Citation

```
@inproceedings{teranishi:2019:naacl,
  title={Decomposed Local Models for Coordinate Structure Parsing},
  author={Teranishi, Hiroki and Shindo, Hiroyuki and Matsumoto, Yuji},
  booktitle={Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year={2019},
  location={Minneapolis, Minnesota},
  publisher={Association for Computational Linguistics},
  url={https://www.aclweb.org/anthology/N19-1343},
  doi={10.18653/v1/N19-1343},
  pages={3394--3403},
}
```

## License

Apache License 2.0
