from typing import Callable
import nltk
from string import digits

# Type alias
SimilarityFn = Callable[[list[str], list[str]], bool]

# Removes numerics in the second item of given pair.
def _remove_numerics(pair: tuple[str, str]):
    return (pair[0],
    ''.join(w.translate({ord(k): None for k in digits}) for w in pair[1]))

_wordlist = nltk.corpus.words.words()
_cmu = nltk.Index(map(_remove_numerics, nltk.corpus.cmudict.entries()))

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

# Get all similar words with the given word in terms of pronunciation.
def _get_word_similars_by_cmu(word: str, similarity_func: SimilarityFn):
    result = set()
    result.add(word)
    pronuns = _cmu[word]
    if len(pronuns) == 0:
        return result
    for curr in _wordlist:
        pr = _cmu[curr]
        if _is_similar(pronuns, pr, similarity_func):
            result.add(curr)
    return result

# Get all similar words with the given words in terms of pronunciation.
def get_similars_by_cmu(tokens: list[str], similarity_func: SimilarityFn):
    return list(map(
        lambda x: (x,
        list(_get_word_similars_by_cmu(x, similarity_func)))
        , tokens))
