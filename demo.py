from typing import Literal
import csv

import numpy as np

from tqdm import tqdm

import pronunciation as P
import evaluation as E
import pipeline

# === [ Configuration ] ===
benchmark_mode: bool = False
benchmark_file = 'inputs.txt'
benchmark_pipeline_replace = True
benchmark_pipeline_restruct = True
benchmark_savefile = 'result_replace_restruct.csv'
benchmark_topk: int = 10

pronunciation_mapping = P.mapping_soundex

semantics_model: pipeline.SemModel = 'sentencebert'

restructure_corenlp_url: str | None = 'http://localhost:9010'
# restructure_corenlp_url: str | None = None

show_top_k = 10
# =========================


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
                result = pipeline.pipeline(
                    ip, benchmark_pipeline_replace, benchmark_pipeline_restruct,
                    pronunciation_mapping, P.simfunc_lcs(rel_thres=0.8, abs_thres=3),
                    semantics_model, restructure_corenlp_url)
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

            pipeline_result = pipeline.pipeline(
                ip, not mode_restruct_only, mode_restruct_only or mode_restruct_with,
                pronunciation_mapping, P.simfunc_lcs(rel_thres=0.8, abs_thres=3),
                semantics_model, restructure_corenlp_url, True)

            print('Results:')
            for idx, res in enumerate(pipeline_result[:show_top_k], start=1):
                print(idx, f'({res[0]:+.3f}) :', res[1])
            print('Done!')
