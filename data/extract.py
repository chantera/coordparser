import argparse

from teras.io.reader import read_tree


def extract_words(tree):
    def _traverse(node, buf):
        assert len(node) > 0
        if len(node) == 2 and isinstance(node[1], str):  # Leaf
            buf.append(node[1])
        else:  # Node
            for child in node[1:]:
                _traverse(child, buf)
    buf = []
    _traverse(tree, buf)
    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='FILE')
    args = parser.parse_args()
    trees = read_tree(args.input)
    for tree in trees:
        if len(tree) == 1:
            tree = tree[0]  # remove redundant brackets
        words = extract_words(tree)
        print(' '.join(words))
