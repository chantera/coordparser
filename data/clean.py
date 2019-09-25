import argparse

from teras.io.reader import read_tree


CC_SEP = [",", ";", ":", "--", "..."]
EXCLUDING_SPAN_LABEL = ["-NONE-", "``", "''"]


def clean_tree(tree, exclusion=None):
    if exclusion is None:
        exclusion = EXCLUDING_SPAN_LABEL
    else:
        assert isinstance(exclusion, (list, tuple))
    words = []

    def _traverse(node, index):
        assert len(node) > 0
        begin = index
        label = node[0]
        if label in exclusion:
            index -= 1
        elif len(node) == 2 and isinstance(node[1], str):  # Leaf
            words.append(node[1])
            span = (begin, index)
        else:  # Node
            conjuncts = []
            conjunct_nodes = []
            for i, child in enumerate(node[1:]):
                child_span = _traverse(child, index)
                if child_span is not None:
                    index = child_span[1] + 1
                    if "COORD" in child[0]:
                        conjuncts.append(child_span)
                        conjunct_nodes.append(i + 1)
                else:
                    node.remove(child)
            if conjuncts:
                insertion = []
                for i in range(1, len(conjuncts)):
                    diff = conjuncts[i][0] - conjuncts[i - 1][1]
                    if diff > 1:  # valid
                        pass
                    elif diff == 1:   # should be fixed
                        if words[conjuncts[i - 1][1]] in CC_SEP:
                            node_index = conjunct_nodes[i - 1]
                            sep = _pop(node[node_index])
                            assert sep[0] in CC_SEP
                            insertion.append((node_index + 1, sep))
                    else:  # invalid
                        raise RuntimeError(
                            "invalid coordination: {}".format(conjuncts))
                for i, sep in reversed(insertion):
                    node.insert(i, sep)
            index -= 1
        span = (begin, index) if begin <= index else None
        return span

    def _pop(node):
        assert not (len(node) == 2 and isinstance(node[1], str))
        if len(node[-1]) == 2 and isinstance(node[-1][1], str):
            popped = node.pop()
            assert len(node) >= 2
        else:
            popped = _pop(node[-1])
            assert len(node[-1]) >= 2
        return popped

    _traverse(tree, index=0)


def serialize(tree):
    def _traverse(node, buf):
        assert len(node) > 0
        buf.append('(')
        buf.append(node[0])
        buf.append(' ')
        for child in node[1:]:
            if isinstance(child, (list, tuple)):
                _traverse(child, buf)
            else:
                buf.append(child)
        buf.append(')')
    buf = []
    _traverse(tree, buf)
    return ''.join(buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='FILE')
    args = parser.parse_args()
    trees = read_tree(args.input)
    for tree in trees:
        if len(tree) == 1:
            tree = tree[0]  # remove redundant brackets
        clean_tree(tree)
        print(serialize(tree))
