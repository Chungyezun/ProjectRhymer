from typing import Callable
import nltk
from string import digits
import re

import misc

# Type alias
SimilarityFn = Callable[[list[str], list[str]], bool]
MappingFn = Callable[[str], list[list[str]]]

# Code by [https://skyjwoo.tistory.com/entry/%EC%9D%B4%EB%A6%84-%EC%9C%A0%EC%82%AC%EB%8F%84-%EA%B5%AC%ED%95%98%EA%B8%B0-soundex-algorithm]
def _encode_soundex(word: str):
    word = word.lower() #소문자로 만들기
    n = len(word)
    vowel_like = ['a', 'e', 'i', 'o', 'u', 'y', 'h', 'w']#drop
    lips = ['b', 'f', 'v', 'p']#1
    others = ['c', 'g','j','k', 'q', 's', 'x', 'z']#2
    dt = ['d', 't']#3
    #l #4
    mn = ['m', 'n']#5
    #r 6
    code = word[0]
    redundant_pat = re.compile(r'([123456])(\1)+') #중복된 패턴 찾기 위함
    
    for i in range(1, n):
        if word[i] in lips:
            code += '1'
        elif word[i] in others:
            code += '2'
        elif word[i] in dt:
            code += '3'
        elif word[i] in 'l':
            code += '4'
        elif word[i] in mn:
            code += '5'
        elif word[i] == 'r':
            code += '6'
        else:
            continue
    while True:
        code_save = code
        code = re.sub(redundant_pat, r'\1', code) #중복된 패턴 하나로 줄이기 44556->456
        
        if code_save == code:
            break
    if len(code) == 1:
        code += '000'
    elif len(code) == 2:
        code += '00'
    elif len(code) == 3:
        code += '0'
    
    return list(code)

def _to_soundex_pair(word: str):
    return (word, _encode_soundex(word))

_wordlist: list[str] = nltk.corpus.words.words()
# Lazy init _cmu.
# _cmu: None | nltk.Index = None
_cmu: misc.LazyWrapper[nltk.Index] = misc.LazyWrapper(lambda: nltk.Index(map(misc.remove_numerics, nltk.corpus.cmudict.entries())))
# Lazy init _soundex.
# _soundex: None | nltk.Index = None
_soundex: misc.LazyWrapper[nltk.Index] = misc.LazyWrapper(lambda: nltk.Index(map(_to_soundex_pair, filter(lambda x: x.isalpha(), _wordlist))))

# Check similarity condition for LCS.
def _check_similarity_lcs(A: list[str], B: list[str], rel_thres=0.5, abs_thres=2) -> bool:
    lcs_len = misc.get_lcs_len(A, B)
    min_len = min(len(A), len(B))
    return (min_len > 0) and (lcs_len >= min_len * rel_thres) and (lcs_len >= abs_thres)

# Return True if rel_sim >= rel_thres and abs_sim >= abs_thres.
def simfunc_lcs(rel_thres=0.5, abs_thres=2) -> SimilarityFn:
    return lambda A, B: _check_similarity_lcs(A, B, rel_thres=rel_thres, abs_thres=abs_thres)

# Internal function to check if any one of As and Bs are similar.
def _is_similar(As: list[list[str]], Bs: list[list[str]], similarity_func: SimilarityFn):
    for A in As:
        for B in Bs:
            if similarity_func(A, B):
                return True
    return False

# Map a word to its pronunciations according to CMU dictionary.
def mapping_cmu(word: str) -> list[list[str]]:
    return _cmu()[word.lower()]

# Map a word to its soundex representation. Returns empty list if it is not an alphabet word.
# Lazy init _soundex.
def mapping_soundex(word: str) -> list[list[str]]:
    if word.isalpha():
        if word in _soundex():
            return _soundex()[word]
        else:
            return [_encode_soundex(word)]
    else:
        return []

# TODO: add [https://github.com/DevTae/PronunciationEvaluator/blob/main/demo.ipynb]
# Support to edit distance: [https://koreascience.kr/article/CFKO201712470015309.pdf]

# Get all similar words for the given word in terms of pronunciation.
def get_similar(word: str, mapping_func: MappingFn, similarity_func: SimilarityFn):
    result = set()
    result.add(word)
    pronuns = mapping_func(word)
    if len(pronuns) == 0:
        return result
    for curr in _wordlist:
        pr = mapping_func(curr)
        if _is_similar(pronuns, pr, similarity_func):
            result.add(curr)
    return result

# Get all similar words for each word in the given sentences in terms of pronunciation.
def get_similars(tokens: list[str], mapping_func: MappingFn, similarity_func: SimilarityFn):
    return list(map(
        lambda x: (x, list(get_similar(x, mapping_func, similarity_func))),
        tokens))

def _calc_rhymeness_lcs(word_a: str, word_b: str, mapping_func: MappingFn) -> float:
    As = mapping_func(word_a)
    Bs = mapping_func(word_b)
    res = 0
    for A in As:
        for B in Bs:
            lcs_len = misc.get_lcs_len(A, B)
            min_len = min(len(A), len(B))
            if min_len > 0:
                res = max(res, lcs_len / min_len)
    return res

def calc_rhymeness_lcs(mapping_func: MappingFn):
    return lambda A, B: _calc_rhymeness_lcs(A, B, mapping_func)
