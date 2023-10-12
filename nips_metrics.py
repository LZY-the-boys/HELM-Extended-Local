import argparse,os
from dataclasses import replace
from typing import List, Optional
import re
from astropy.table import Table
import pandas as pd
from functools import reduce
import numpy as np
from tabulate import tabulate

def main(args):

    acc_data = calc_acc(args)
    robust_data = calc_robustness(args)
    fair_data = calc_fairness(args)
    bias_data = calc_bias(args)
    
    data = reduce(lambda x, y: 
        pd.merge(x, y, on = 'Model/adapter',how='outer'), 
        [acc_data,robust_data, fair_data, bias_data],
    )
    data['score'] = data.apply(lambda row: np.mean(data['acc-winrate']+data['robust-winrate']+data['fair-winrate']+data['bias-winrate']), axis=1)

    data.to_csv(os.path.join(args.output_path, f'{args.suite}.csv'))
    data.to_excel(os.path.join(args.output_path, f'{args.suite}.xlsx'))

    markdown_table = tabulate(data, headers='keys', tablefmt='pipe')
    print(markdown_table)
    print(markdown_table, file=open(os.path.join(args.output_path, f'{args.suite}.md'), 'w'))

def calc_winrate(data):
    # TODO: 相同的值会随机分配不同的winrate
    win_rates_per_row = [[] for _ in data.index]
    for col in data.columns:
        # don't rank single model
        if len(data[col].dropna()) < 2:
            continue
        sorted_df = data.sort_values(col)
        for i, (index, row) in enumerate(sorted_df.iterrows()):
            win_rate = i / (len(sorted_df) - 1)  # normalize to [0, 1]
            win_rates_per_row[index].append(win_rate)

    aggregate_win_rates = []
    for win_rates in win_rates_per_row:
        aggregate = np.mean(win_rates)
        aggregate_win_rates.append(aggregate)
    return aggregate_win_rates

def calc_acc(args):
    path1 = os.path.join(args.latex_dir,'core_scenarios_accuracy.tex')
    path2 = os.path.join(args.latex_dir,'targeted_evaluations_accuracy.tex')
    data = pd.merge(
        Table.read(path1).to_pandas(), 
        Table.read(path2).to_pandas(), 
        how='left',
    )
    data=data.drop(['Mean win rate'],axis=1)
    # winrate 需要重新计算
    data['acc-winrate']=calc_winrate(data)
    return data

def calc_bias(args):
    path1 = os.path.join(args.latex_dir,'core_scenarios_bias.tex')
    data = Table.read(path1).to_pandas()
    data['bias-winrate']=data['Mean win rate']
    data=data.drop(['Mean win rate'],axis=1)
    return data

def calc_fairness(args):
    path1 = os.path.join(args.latex_dir,'core_scenarios_fairness.tex')
    data = Table.read(path1).to_pandas()
    data['fair-winrate']=data['Mean win rate']
    data=data.drop(['Mean win rate'],axis=1)
    return data

def calc_robustness(args):
    path1 = os.path.join(args.latex_dir,'core_scenarios_robustness.tex')
    data = Table.read(path1).to_pandas()
    data['robust-winrate']=data['Mean win rate']
    data=data.drop(['Mean win rate'],axis=1)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", type=str, help="the suite name", default="tmp"
    )
    parser.add_argument(
        "--output-path", type=str, help="the output dir", default="/home/lzy/HELM-Extended-Local/"
    )

    args = parser.parse_args()
    args.latex_dir = os.path.join(
        'benchmark_output/runs',
        os.path.join(args.suite, 'groups/latex')
    )
    main(args)