from typing import Literal
import nltk
import math
import csv

import numpy as np

from tqdm import tqdm

import pronunciation as P
import semantics_first as Sf
import semantics as S
import restructurer as R
import evaluation as E

# === [ Configuration ] ===
benchmark_mode: bool = True
benchmark_file = 'inputs.txt'
benchmark_pipeline_replace = True
benchmark_pipeline_restruct = False
benchmark_savefile = 'result.csv'
benchmark_topk: int = 10

pronunciation_mapping = P.mapping_soundex

SemModel = Literal['wordnet'] | Literal['bert'] | Literal['sentencebert']
semantics_model: SemModel = 'sentencebert'

restructure_corenlp_url: str | None = 'http://localhost:9010'
# restructure_corenlp_url: str | None = None

show_top_k = 10
# =========================

def pipeline(original_text: str, pipeline_replace: bool=True, pipeline_restruct: bool=False, verbose: bool=False) -> list[tuple[float, str]]:
    inter_results: list[str] = []
    final_results: list[str] = []

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
            P.simfunc_lcs(rel_thres=0.8, abs_thres=3)
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
        step4_sets = [set(R.restructure(text, restructure_corenlp_url)) for text in step4_texts]
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

if __name__ == '__main__':
    if benchmark_mode:
        print('Benchmark mode')
        print('Inputs from:', benchmark_file)
        fieldnames = ['idx', 'orig_rhyme', 'num_outs', 'max', 'topk_min', 'topk_avg', 'topk_med', 'min', 'avg', 'med']
        with open(benchmark_file, 'r', encoding='utf-8') as binput, open(benchmark_savefile, 'w', encoding='utf-8', newline='') as boutput:
            writer = csv.DictWriter(boutput, fieldnames=fieldnames)
            writer.writeheader()
            for idx, line in enumerate(tqdm(binput.readlines(), desc='Lines processed')):
                ip = line.strip()
                if ip == '':
                    continue
                rhymeness_original = E.eval_rhyme_score(ip)
                result = pipeline(ip, benchmark_pipeline_replace, benchmark_pipeline_restruct)
                result_scores = [p[0] for p in result]
                writer.writerow({
                    'idx': idx, 'orig_rhyme': rhymeness_original, 'num_outs': len(result),
                    'max': np.max(result_scores),
                    'topk_min': np.min(result_scores[:benchmark_topk]),
                    'topk_avg': np.mean(result_scores[:benchmark_topk]),
                    'topk_med': np.median(result_scores[:benchmark_topk]),
                    'min': np.min(result_scores),
                    'avg': np.mean(result_scores),
                    'med': np.median(result_scores),
                })
        print('Benchmark is Done!')
    else:
        print('Type nothing to stop loop.')
        print('Begin with # to do only restructure')
        print('Begin with ! to do with restructure')
        while True:
            ip = input('> ')
            if ip == '':
                break
            # Pipeline
            rhymeness_original = E.eval_rhyme_score(ip)
            print(f'Original rhymeness: {rhymeness_original:.3f}')

            mode_restruct_only = ip[0] == '#'
            mode_restruct_with = ip[0] == '!'

            if mode_restruct_only or mode_restruct_with:
                ip = ip[1:]

            pipeline_result = pipeline(ip, not mode_restruct_only, mode_restruct_only or mode_restruct_with, True)

            print('Results:')
            for idx, res in enumerate(pipeline_result[:show_top_k], start=1):
                print(idx, f'({res[0]:+.3f}) :', res[1])
            print('Done!')
