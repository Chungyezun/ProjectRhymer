import csv
import numpy as np

target_fields = ['max', 'topk_avg', 'topk_med', 'num_outs']

def print_analysis(filename: str):
    print('File:', filename)
    with open(filename, 'r', encoding='utf-8', newline='') as rf:
        creader = csv.DictReader(rf)
        ent: dict[list[float]] = dict()
        for field in target_fields:
            ent[field] = []
        for row in creader:
            for field in target_fields:
                ent[field].append(float(row[field]))
        for field in target_fields:
            print(f'  {field: >10} : {np.mean(ent[field]):.3f}')

if __name__ == '__main__':
    print_analysis('result_replace_only.csv')
    print_analysis('result_restruct_only.csv')
    print_analysis('result_replace_restruct.csv')
