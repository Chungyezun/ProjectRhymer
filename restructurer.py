import nltk
import pprint
import itertools
import functools
import math

# NOTE: Use CoreNLP or MaltParser, StanfordParser, SpaCy

# In order to use CoreNLP, You should download the CoreNLP from the website
# [https://stanfordnlp.github.io/CoreNLP/index.html#download]
# Instead one can use:
# java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9010 -timeout 15000
# one the same directory with jar
_nlpsv = nltk.parse.corenlp.CoreNLPServer('./artifacts/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7.jar', './artifacts/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7-models.jar')

_cnlp: None | nltk.parse.CoreNLPParser = None
_wnlm = nltk.WordNetLemmatizer()

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
    global _cnlp
    res = [sentence]
    res_per_sent = []
    for stree in _cnlp.parse_text(sentence):
        if not isinstance(stree, nltk.Tree):
            res_per_sent.append([])
            continue
        # stree.pretty_print()
        # work on global S
        # assmue ROOT - S - ... structure
        processed = False
        try:
            if len(stree) > 0:
                subtree = stree[0]
                if isinstance(subtree, nltk.Tree) and subtree.label() == 'S':
                    # Work only on simple sentences
                    # if len(subtree) < 2 or not isinstance(subtree[0], nltk.Tree) or subtree[0].label() != 'NP' or not isinstance(subtree[1], nltk.Tree) or subtree[1].label() != 'VP':
                    #     continue
                    if len(subtree) < 2:
                        continue
                    # If VP (MD? VB_ NP PP)
                    # Put PP out
                    # Test if there is a vp
                    vp_idx = None
                    for i, ch in enumerate(subtree):
                        if isinstance(ch, nltk.Tree) and ch.label() == 'VP':
                            vp_idx = i
                            break
                    if vp_idx is None:
                        continue
                    # Inversion: Forwarded word
                    inv_fw = ''
                    # Inversion: Replaced verb
                    inv_rp = None
                    # Test if found VP is simple
                    vp_simple = True
                    vb_idx = None
                    inv_pp = None
                    inv_modal = False
                    vp_words = subtree[vp_idx].flatten()
                    if 'not' in vp_words or "n't" in vp_words:
                        # Have complex structure, and sometimes have ambiguity in PP. (I do not like it.)
                        continue
                    # We have modal...
                    inv_mainvp = None
                    if isinstance(subtree[vp_idx][0], nltk.Tree) and subtree[vp_idx][0].label() == 'MD':
                        # Assume VP(MD VP)
                        inv_fw = subtree[vp_idx][0].flatten()[0]
                        inv_modal = True
                        inv_mainvp = subtree[vp_idx][1]
                    else:
                        inv_mainvp = subtree[vp_idx]
                    for i, ch in enumerate(inv_mainvp):
                        if isinstance(ch, nltk.Tree):
                            if ch.label().startswith('VB'):
                                if vb_idx is not None:
                                    vp_simple = False
                                    break
                                vb_idx = i
                            elif ch.label() == 'PP':
                                if inv_pp is not None:
                                    vp_simple = False
                                    break
                                inv_pp = ch.copy()
                            elif ch.label() == 'MD':
                                vp_simple = False
                                break
                    if not vp_simple or vb_idx is None or inv_pp is None:
                        continue
                    # Preprocessing
                    if not inv_modal:
                        v_tense = 'past' if inv_mainvp[vb_idx].label() == 'VBD' else 'present'
                        v_trd = '3rd' if inv_mainvp[vb_idx].label() == 'VBZ' else 'no-3rd'
                        v_str = inv_mainvp[vb_idx].flatten()[0]
                        vb_base_form = _wnlm.lemmatize(v_str, pos='v')
                        # print(' v base:', vb_base_form)
                        if vb_base_form == 'be':
                            inv_fw = v_str
                            inv_rp = None
                        else:
                            inv_rp = vb_base_form
                            if v_tense == 'past':
                                inv_fw = 'did'
                            else:
                                inv_fw = 'does' if v_trd == '3rd' else 'do'
                    else:
                        inv_rp = inv_mainvp[vb_idx].flatten()[0]
                    # Perform actual inversion process.
                    invc_chs = [inv_pp, inv_fw]
                    for i in range(len(subtree)):
                        if i != vp_idx:
                            invc_chs.append(subtree[i])
                        else:
                            invc_vpchs = []
                            for i, ch in enumerate(inv_mainvp):
                                if i == vb_idx:
                                    if inv_rp is not None:
                                        invc_vpchs.append(inv_rp)
                                else:
                                    if not (isinstance(ch, nltk.Tree) and ch.label() == 'PP'):
                                        invc_vpchs.append(ch)
                            invc_chs.append(nltk.Tree('VP', invc_vpchs))
                    invc_new_s = nltk.Tree('S', invc_chs)
                    processed = True
                    res_per_sent.append([' '.join(stree.leaves()), ' '.join(invc_new_s.flatten())])
        finally:
            if not processed:
                res_per_sent.append([' '.join(stree.leaves())])
    total_possiblity = itertools.product(*res_per_sent)
    str_res = []
    str_res.extend([' '.join(oc) for oc in total_possiblity])
    res.extend(str_res)
    final_result = list(set(filter(lambda x: len(x) > 0, res)))
    return final_result

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
        if not isinstance(stree, nltk.Tree):
            res.append([])
            continue
        # stree.pretty_print()
        # Our target:
        # ADVP (freely move along siblings)
        # TODO:
        # , PP , -> put first, last
        # PP <-> PP? -> interchangable
        try:
            dup_trees = tree_shaker(stree)
            res.append([' '.join(t.leaves()) for t in dup_trees])
        except:
            res.append([' '.join(stree.leaves())])
    total_possiblity = itertools.product(*res)
    str_res = []
    str_res.extend([' '.join(oc) for oc in total_possiblity])
    final_result = list(set(filter(lambda x: len(x) > 0, str_res)))
    # print(final_result)
    return final_result

# Get possible reorder candidates,
# need to filter out some incorrectly grammared and different meaning sentences.
def restructure(sentence: str, external_url: str | None = None) -> list[str]:
    global _cnlp
    if external_url is None:
        with _nlpsv:
            _cnlp = nltk.parse.CoreNLPParser(url=_nlpsv.url)
            res = reorder_inversion(sentence)
            res = list(set(sent for target in res for sent in reorder_adverb(target)))
            _cnlp = None
    else:
        _cnlp = nltk.parse.CoreNLPParser(url=external_url)
        res = reorder_inversion(sentence)
        res = list(set(sent for target in res for sent in reorder_adverb(target)))
        _cnlp = None
    return res

if __name__ == '__main__':
    ip = '.'
    while True:
        ip = input('> ')
        if ip == '':
            break
        print(restructure(ip, 'http://localhost:9010'))
# analyze_sent()
