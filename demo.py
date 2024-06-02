from typing import Literal
import nltk
import math

import pronunciation as P
import semantics_first as Sf
import semantics as S
import restructurer as R
import evaluation as E

# === [ Configuration ] ===
pronunciation_mapping = P.mapping_soundex

SemModel = Literal['wordnet'] | Literal['bert'] | Literal['sentencebert']
semantics_model: SemModel = 'sentencebert'

# restructure_corenlp_url: str | None = 'http://localhost:9010'
restructure_corenlp_url: str | None = None

show_top_k = 10
# =========================

if __name__ == '__main__':
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
        
        final_results: list[str] = []
        if mode_restruct_only:
            # 4. [R] step
            print('Pipeline ... 4')
            mod_ip = ip[1:]
            step4_reformed = set()
            step4_reformed.add(mod_ip)
            step4_sets = set(R.restructure(mod_ip, restructure_corenlp_url))
            step4_reformed.union(step4_sets)
            final_results = list(step4_reformed)
        else:
            if mode_restruct_with:
                ip = ip[1:]

            # 1. Tokenize
            print('Pipeline ... 1')
            original_tokens = nltk.word_tokenize(ip)
            # original_tokens = [w.lower() for w in original_tokens]

            # 2. [P] step
            print('Pipeline ... 2')
            step2_tokens = P.get_similars(
                original_tokens,
                pronunciation_mapping,
                P.simfunc_lcs(rel_thres=0.8, abs_thres=3)
                )
            step2_ss = math.prod([len(tc[1]) for tc in step2_tokens])
            print('  Search space:', step2_ss)
            
            # 3. [S] step
            print('Pipeline ... 3')
            step3_tokens = S.recommend_semantics(step2_tokens, model=semantics_model, threshold=0.4 if semantics_model == 'sentencebert' else 0.8)
            print('  # of results:', len(step3_tokens))

            if mode_restruct_with:
                # This step takes too much time...
                # 4. [R] step
                print('Pipeline ... 4')
                step4_texts = [' '.join(text) for text in step3_tokens]
                step4_texts.append(ip)
                step4_reformed = set()
                step4_sets = [set(R.restructure(text, restructure_corenlp_url)) for text in step4_texts]
                for sset in step4_sets:
                    step4_reformed.update(sset)
                final_results = list(step4_reformed)
                print('  # of results:', len(final_results))
            else:
                final_results = [' '.join(text) for text in step3_tokens]
                final_results.append(ip)
            
        # 0. Eval and Show
        print('Evaluating ...')
        step0_evals: list[tuple[float, str]] = []
        for idx, rtext in enumerate(final_results, start=1):
            rhymeness = E.eval_rhyme_score(rtext)
            rdiff = rhymeness - rhymeness_original
            step0_evals.append((rdiff, rtext))
        step0_evals.sort(key=lambda x: x[0], reverse=True)

        print('Results:')
        for idx, res in enumerate(step0_evals[:show_top_k], start=1):
            print(idx, f'({res[0]:+.3f}) :', res[1])
        print('Done!')
