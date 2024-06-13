import csv
import itertools

import numpy as np

from tqdm import tqdm

import pronunciation as P
import evaluation as E
import pipeline

# === [ Configuration ] ===
bm_input_file = 'inputs.txt'
bm_pipeline_replace = True
bm_pipeline_restruct = True
bm_report_format = 'benchmark_{}.csv'
bm_topk: int = 10

pronunciation_mapping = P.mapping_soundex
pronunciation_simfunc = P.simfunc_lcs()

semantics_model: pipeline.SemModel = 'sentencebert'

restructure_corenlp_url: str | None = 'http://localhost:9010'
# =========================

def run_benchmark(section_name: str):
    fieldnames = ['idx', 'orig_rhyme', 'num_outs', 'max', 'topk_min', 'topk_avg', 'topk_med', 'min', 'avg', 'med']
    with open(bm_input_file, 'r', encoding='utf-8') as binput, open(bm_report_format.format(section_name), 'w', encoding='utf-8', newline='') as boutput:
        writer = csv.DictWriter(boutput, fieldnames=fieldnames)
        writer.writeheader()
        for idx, line in enumerate(tqdm(binput.readlines(), desc='Lines processed')):
            ip = line.strip()
            if ip == '':
                continue
            rhymeness_original = E.eval_rhyme_score(ip)
            result = pipeline.pipeline(ip, bm_pipeline_replace, bm_pipeline_restruct, pronunciation_mapping, pronunciation_simfunc, semantics_model, restructure_corenlp_url)
            result_scores = [p[0] for p in result]
            writer.writerow({
                'idx': idx, 'orig_rhyme': rhymeness_original, 'num_outs': len(result),
                'max': np.max(result_scores),
                'topk_min': np.min(result_scores[:bm_topk]),
                'topk_avg': np.mean(result_scores[:bm_topk]),
                'topk_med': np.median(result_scores[:bm_topk]),
                'min': np.min(result_scores),
                'avg': np.mean(result_scores),
                'med': np.median(result_scores),
            })

def run_all_benchmarks():
    global bm_pipeline_replace, bm_pipeline_restruct, pronunciation_mapping, pronunciation_simfunc, semantics_model

    print('Inputs from:', bm_input_file)
    arg_pipesteps: list[tuple[bool, bool]] = [(True, False), (False, True), (True, True)]
    arg_pron_map_thres: list[tuple[P.MappingFn, P.SimilarityFn, str]] = [
        (P.mapping_soundex, P.simfunc_lcs(rel_thres=0.8, abs_thres=3), 'sdx'),
        (P.mapping_cmu, P.simfunc_lcs(rel_thres=0.7, abs_thres=2), 'cmu')]
    arg_sem_model: list[pipeline.SemModel] = ['wordnet', 'bert', 'sentencebert']

    for args in itertools.product(arg_pipesteps, arg_pron_map_thres, arg_sem_model):
        bm_pipeline_replace = args[0][0]
        bm_pipeline_restruct = args[0][1]
        pronunciation_mapping = args[1][0]
        pronunciation_simfunc = args[1][1]
        semantics_model = args[2]

        section_name = '_'.join([
            'repl' if bm_pipeline_replace else 'norepl',
            'rest' if bm_pipeline_restruct else 'norest',
            args[1][2],
            args[2]
        ])
        print('Run Section:', section_name)
        run_benchmark(section_name)
    print('Done benchmarks!')

if __name__ == '__main__':
    run_all_benchmarks()
