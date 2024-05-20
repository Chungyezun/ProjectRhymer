from typing import Callable
import nltk

import misc

# Function Types
RhymeMetricFn = Callable[[], float]
SemanticMetricFn = Callable[[], float]

_cmu: misc.LazyWrapper[nltk.Index] = misc.LazyWrapper(lambda: nltk.Index(map(misc.remove_numerics, nltk.corpus.cmudict.entries())))

def rmetric_cmu(orig: list[str], target: list[str]):
    def convert_to_cmu(token: str) -> str:
        if token.lower() not in _cmu():
            print('(Token:', token, ' not found on CMU)')
            return token
        else:
            # TODO: take multiple candidates into consideration.
            return _cmu()[token.lower()][0]
    original_pron_str = ''.join(map(convert_to_cmu, orig))
    target_pron_str = ''.join(map(convert_to_cmu, target))
    lcs_len = misc.get_lcs_len(list(original_pron_str), list(target_pron_str))
    print('For "{}" <-> "{}" = "{}" - "{}": {} <{}> {}'.format(' '.join(orig), ' '.join(target), original_pron_str, target_pron_str, len(original_pron_str), lcs_len, len(target_pron_str)))
    return lcs_len / min(len(original_pron_str), len(target_pron_str))

def smetric_zero():
    return 0.0

# Evaluate Rhyme Score for target generated sentence, given the original sentence.
def eval_rhyme_score(original: list[str], target: list[str], metric_func: RhymeMetricFn) -> float:
    return metric_func(original, target)

# Evaluate Semantic Score for target generated sentence, given the original sentence.
def eval_sem_score(original: list[str], target: list[str], metric_func: SemanticMetricFn) -> float:
    return metric_func(original, target)
