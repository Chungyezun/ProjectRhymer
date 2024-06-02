from typing import Callable
import nltk
import itertools

import misc

_cmu: misc.LazyWrapper[nltk.Index] = misc.LazyWrapper(lambda: nltk.Index(map(misc.remove_numerics, nltk.corpus.cmudict.entries())))

def _convert_to_cmu(token: str) -> list[list[str]]:
    # if token.lower() not in _cmu():
    #     print('(Token:', token, ' not found on CMU)')
    return _cmu()[token.lower()]

# TODO: Use nltk.edit_distance?
def _calc_rhymeness_lcs(word_a: str, word_b: str) -> float:
    As = _convert_to_cmu(word_a)
    Bs = _convert_to_cmu(word_b)
    res = 0
    for A in As:
        for B in Bs:
            lcs_len = misc.get_lcs_len(A, B)
            min_len = min(len(A), len(B))
            if min_len > 0:
                res = max(res, lcs_len / min_len)
    return res

def _calc_rhymeness_edit(word_a: str, word_b: str) -> float:
    As = _convert_to_cmu(word_a)
    Bs = _convert_to_cmu(word_b)
    res = 0
    for A in As:
        for B in Bs:
            edit_dist = nltk.edit_distance(A, B)
            res = max(res, 1 / (edit_dist + 1))
    return res

def rmetric_cmu_w2w(word_a: str, word_b: str) -> float:
    # TODO: how about using activation function on this?
    return _calc_rhymeness_lcs(word_a, word_b)

def rmetric_cmu_sent(sentence: str) -> float:
    words = nltk.word_tokenize(sentence)
    words = list(filter(lambda x: x != "", words))
    sent_rhymeness = [rmetric_cmu_w2w(wa[1], wb[1]) for wa in enumerate(words) for wb in enumerate(words) if wa[0] != wb[0]]
    # TODO: use mean?
    return sum(sent_rhymeness)

def _positional_weight(pos_a: int, pos_b: int, sent_len_a: int, sent_len_b: int) -> float:
    def get_relative_pos(idx: int, length: int) -> float:
        return (idx + 0.5) / length
    return 1 - abs(get_relative_pos(pos_a, sent_len_a) - get_relative_pos(pos_b, sent_len_b))

def rmetric_cmu_s2s(sent_a: str, sent_b: str) -> float:
    words_a = nltk.word_tokenize(sent_a)
    words_a = list(filter(lambda x: x != "", words_a))
    words_b = nltk.word_tokenize(sent_b)
    words_b = list(filter(lambda x: x != "", words_b))
    len_a = len(words_a)
    len_b = len(words_b)
    rs_arr = []
    for idx_a, word_a in enumerate(words_a):
        rs_arr.append([
            (rmetric_cmu_w2w(word_a, word_b) * _positional_weight(idx_a, idx_b, len_a, len_b))
            for idx_b, word_b in enumerate(words_b)])
    # TODO: try different aggregate function?
    return sum(max(rs_single) for rs_single in rs_arr)

# sent-sent rhymeness
# in-sent rhymeness
# Evaluate Rhyme Score for target generated sentence
def eval_rhyme_score(text: str) -> float:
    sents = nltk.sent_tokenize(text)
    sents = list(filter(lambda x: x != "", sents))
    # Calculate in-sentence rhymeness
    rs_insent = [rmetric_cmu_sent(sent) for sent in sents]
    # Calculate sent-sent rhymeness
    rs_s2s = [rmetric_cmu_s2s(sa[1], sb[1]) for sa in enumerate(sents) for sb in enumerate(sents) if sa[0] != sb[0]]\
    # TODO: Sum?
    rhymeness = sum(rs_insent) + sum(rs_s2s)
    return rhymeness 

if __name__ == '__main__':
    while True:
        ip = input('> ')
        if ip == '':
            break
        print(eval_rhyme_score(ip))
