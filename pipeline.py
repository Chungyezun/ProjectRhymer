from typing import Literal
import math
import nltk

import pronunciation as P
import semantics_first as Sf
import semantics as S
import restructurer as R
import evaluation as E

SemModel = Literal['wordnet'] | Literal['bert'] | Literal['sentencebert']

def pipeline(
        original_text: str,
        pipeline_replace: bool, pipeline_restruct: bool,
        pronunciation_mapping: P.MappingFn,
        pronunciation_simfunc: P.SimilarityFn,
        semantics_model: SemModel,
        corenlp_url: None | str,
        verbose: bool=False) -> list[tuple[float, str]]:
    inter_results: list[str] = []
    final_results: list[str] = []

    rhymeness_original = E.eval_rhyme_score(original_text)

    if pipeline_replace:
        # 1. Tokenize
        if verbose: print('Pipeline ... 1')
        original_tokens = nltk.word_tokenize(original_text)
        # original_tokens = [w.lower() for w in original_tokens]

        # 2. [P] step
        if verbose: print('Pipeline ... 2')
        step2_tokens = P.get_similars(
            original_tokens,
            pronunciation_mapping,
            pronunciation_simfunc
            )
        step2_ss = math.prod([len(tc[1]) for tc in step2_tokens])
        if verbose: print('  Search space:', step2_ss)
        
        # 3. [S] step
        if verbose: print('Pipeline ... 3')
        step3_tokens = S.recommend_semantics(step2_tokens, model=semantics_model, threshold=0.4 if semantics_model == 'sentencebert' else 0.8)
        if verbose: print('  # of results:', len(step3_tokens))
        inter_results = [' '.join(text) for text in step3_tokens]
        inter_results.append(original_text)
    else:
        inter_results = [original_text]

    # print(inter_results)
    
    if pipeline_restruct:
        # This step takes too much time...
        # 4. [R] step
        if verbose: print('Pipeline ... 4')
        step4_texts = inter_results
        step4_reformed = set()
        step4_sets = [set(R.restructure(text, corenlp_url)) for text in step4_texts]
        for sset in step4_sets:
            step4_reformed.update(sset)
        final_results = list(step4_reformed)
        if verbose: print('  # of results:', len(final_results))
    else:
        final_results = inter_results

    # print(final_results)

    # 0. Eval
    if verbose: print('Evaluating ...')
    step0_evals: list[tuple[float, str]] = []
    for rtext in final_results:
        rhymeness = E.eval_rhyme_score(rtext)
        rdiff = rhymeness - rhymeness_original
        step0_evals.append((rdiff, rtext))
    step0_evals.sort(key=lambda x: x[0], reverse=True)
    return step0_evals
