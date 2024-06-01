import nltk
import pprint
import itertools
import functools
import math

# NOTE: Use CoreNLP or MaltParser, StanfordParser, SpaCy

# In order to use CoreNLP, You should download the CoreNLP from the website
# [https://stanfordnlp.github.io/CoreNLP/index.html#download]
_nlpsv = nltk.parse.corenlp.CoreNLPServer('./artifacts/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7.jar', './artifacts/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7-models.jar')
_cnlp: None | nltk.parse.CoreNLPParser = None

def analyze_sent():
    with _nlpsv:
        cnlp_parser = nltk.parse.CoreNLPParser(url=_nlpsv.url)
        while True:
            ip = input('> ')
            print('Got input', ip)
            if ip == "":
                break
            print('  Parsing...')
            for tree in cnlp_parser.parse_text(ip):
                tree.pretty_print()
            print('  Parsing... Done')

# Discover every possible applications of inversions.
def reorder_inversion(sentence: str) -> list[str]:
    # Hardcode some basic inversion rules.
    return [sentence]

# Discover some possible reordering of adverbs.
# maybe use dependency parser and put adverbs around the word
def reorder_adverb(sentence: str) -> list[str]:
    # Try to locate some adverbs and shuffle them around.
    global _cnlp
    def tree_shaker(tree: nltk.Tree) -> list[nltk.Tree]:
        if tree.height() <= 2:
            return [tree]
        advp_indexes = []
        for i in range(len(tree)):
            if isinstance(tree[i], nltk.Tree) and tree[i].label() == 'ADVP':
                advp_indexes.append(i)
        advp_cnt = len(advp_indexes)
        if advp_cnt == 0:
            outcomes = [tree_shaker(child) if isinstance(child, nltk.Tree) else [child] for child in tree]
            total_possiblity = itertools.product(*outcomes)
            return [nltk.Tree(tree.label(), outcome) for outcome in total_possiblity]
        else:
            res = []
            last_dot = (isinstance(tree[-1], nltk.Tree) and tree[-1].label() == '.')
            # Represents tree[i:j] = [i, j)
            tree_intervals = []
            tree_intervals.append((0, advp_indexes[0]))
            for i in range(advp_cnt - 1):
                tree_intervals.append((advp_indexes[i] + 1, advp_indexes[i + 1]))
            if last_dot:
                tree_intervals.append((advp_indexes[-1] + 1, len(tree) - 1))
            else:
                tree_intervals.append((advp_indexes[-1] + 1, len(tree)))
            tree_intervals = list(filter(lambda x: x[0] < x[1], tree_intervals))
            tree_intervals.insert(0, (0, 0))
            # print('tree_intervals:', tree_intervals)
            # print('advp:', advp_indexes)
            for ips in itertools.combinations_with_replacement(range(len(tree_intervals)), advp_cnt):
                for perms in itertools.permutations(range(advp_cnt)):
                    cpt = [[] for _ in range(len(tree_intervals))]
                    for ip, aidx in zip(ips, perms):
                        cpt[ip].append(tree[advp_indexes[aidx]])
                    outcome = []
                    # print(' outcome:', ips, perms, cpt)
                    for a, b in zip(tree_intervals, cpt):
                        outcome.extend(tree[a[0]:a[1]])
                        outcome.extend(b)
                    if last_dot:
                        outcome.extend(tree[-1])
                    # Recursively extend the outcome
                    ext_outcomes = [
                        (tree_shaker(child) if isinstance(child, nltk.Tree) and child.label() != 'ADVP' and child.label() not in ',.' else [child])
                        for child in outcome
                    ]
                    total_possiblity = itertools.product(*ext_outcomes)
                    res.extend([nltk.Tree(tree.label(), oc) for oc in total_possiblity])
            # print('res:', res)
            return res
    # Refer to [https://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html]
    # for the tag info
    res = []
    for stree in _cnlp.parse_text(sentence):
        stree.pretty_print()
        # Our target:
        # ADVP (freely move along siblings)
        # , PP , -> put first, last
        # PP <-> PP? -> interchangable
        dup_trees = tree_shaker(stree)
        res.append([' '.join(t.leaves()) for t in dup_trees])
    total_possiblity = itertools.product(*res)
    str_res = []
    str_res.extend([' '.join(oc) for oc in total_possiblity])
    return str_res

# Get possible reorder candidates,
# need to filter out some incorrectly grammared and different meaning sentences.
def restructure(sentence: str) -> list[str]:
    global _cnlp
    with _nlpsv:
        _cnlp = nltk.parse.CoreNLPParser(url=_nlpsv.url)
        res = reorder_inversion(sentence)
        res = [sent for target in res for sent in reorder_adverb(target)]
        _cnlp = None
    return res

print(restructure('I really hope that one will be black obviously.'))
# analyze_sent()
