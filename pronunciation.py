import nltk
from string import digits

def _remove_numerics(pair):
    return (pair[0],
    [w.translate({ord(k): None for k in digits}) for w in pair[1]])

_wordlist = nltk.corpus.words.words()
_cmu = nltk.Index(map(_remove_numerics, nltk.corpus.cmudict.entries()))

# Return True if rel_sim >= rel_thres and abs_sim >= abs_thres
def check_similarity_lcs(A, B, rel_thres=0.5, abs_thres=2):
    
    return False

def simfunc(func, **kwargs):
    return lambda A, B: func(A, B, **kwargs)

def _is_similar(As, Bs, similarity_func):
    for A in As:
        for B in Bs:
            if similarity_func(A, B):
                return True
    return False

def _get_word_similars_by_cmu(word, similarity_func):
    result = set()
    result.add(word)
    pronuns = _cmu[word]
    if len(pronuns) == 0:
        return result
    for curr in _wordlist:
        pr = _cmu[curr]
        if _is_similar(pronuns, pr):
            result.add(curr)
    return result

def get_similars_by_cmu(tokens, similarity_func):
    return list(map(
        lambda x: (x,
        list(_get_word_similars_by_cmu(x, similarity_func)))
        , tokens))
