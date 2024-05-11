from typing import Callable
import nltk
from string import digits
import re

# Type alias
SimilarityFn = Callable[[list[str], list[str]], bool]
MappingFn = Callable[[str], list[str]]

# Removes numerics in the second item of given pair.
def _remove_numerics(pair: tuple[str, str]):
    return (pair[0],
    ''.join(w.translate({ord(k): None for k in digits}) for w in pair[1]))

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
    
    return code

def _to_soundex_pair(word: str):
    return (word, _encode_soundex(word))

_wordlist = nltk.corpus.words.words()
# Lazy init _cmu.
_cmu: None | nltk.Index = None
# Lazy init _soundex.
_soundex: None | nltk.Index = None

# Perform actual LCS algorithm.
def _lcs_len(A: list[str], B: list[str]) -> int:
    la = len(A)
    lb = len(B)
    dp = [[0 for i in range(lb + 1)] for j in range(la + 1)]
    for i in range(la):
        for j in range(lb):
            dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            if A[i].lower() == B[j].lower():
                dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j] + 1)
    return dp[la][lb]

# Check similarity condition for LCS.
def _check_similarity_lcs(A: list[str], B: list[str], rel_thres=0.5, abs_thres=2) -> bool:
    lcs_len = _lcs_len(A, B)
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
# Lazy init _cmu.
def mapping_cmu(word: str) -> list[str]:
    global _cmu
    if _cmu is None:
        _cmu = nltk.Index(map(_remove_numerics, nltk.corpus.cmudict.entries()))
    return _cmu[word]

# Map a word to its soundex representation. Returns empty list if it is not an alphabet word.
# Lazy init _soundex.
def mapping_soundex(word: str) -> list[str]:
    global _soundex
    if _soundex is None:
        _soundex = nltk.Index(map(_to_soundex_pair, filter(lambda x: x.isalpha(), _wordlist)))
    if word.isalpha():
        if word in _soundex:
            return _soundex[word]
        else:
            return [_encode_soundex(word)]
    else:
        return []

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
        lambda x: (x, list(get_similars(x, mapping_func, similarity_func))),
        tokens))
