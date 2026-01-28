#!/usr/bin/env python3
"""
Generate table-only LaTeX code for scalp EEG model statistics.
Shows statistical comparisons between models and benchmarks.
"""

import pandas as pd
import numpy as np
from config import Config
from os.path import join as ospj

prodatapath = Config.deal(['prodatapath'])

def main():
    # Read the CSV
    df = pd.read_csv(ospj(prodatapath,'scalp_model_stats_table.csv'))
    
    # Custom model name mapping
    model_names = {
        'DynaSD-LiNDDA-256-10min-32-final': 'LiNDDA-32',
        'DynaSD-LiNDDA-256-10min-54-final': 'LiNDDA-54',
        'DynaSD-NDD-256-0.01-10min-final': 'NDD',
        'kaggle': 'Kaggle',
        'ramses': 'RAMSES',
        'sparcnet': 'SparcNet'
    }
    
    # Analysis set labels
    analysis_labels = {
        'old': 'Test Set',
        'new': 'Validation Set',
        'old+new': 'Combined'
    }
    
    # Apply mappings
    df['model_short'] = df['model'].map(model_names).fillna(df['model'])
    df['benchmark_short'] = df['benchmark_model'].map(model_names).fillna(df['benchmark_model'])
    df['analysis_label'] = df['analysis_set'].map(analysis_labels).fillna(df['analysis_set'])
    
    # Format p-values
    def format_pvalue(val):
        if pd.isna(val):
            return '--'
        if val < 0.001:
            return f'{val:.2e}'
        else:
            return f'{val:.3f}'
    
    df['p_val_fmt'] = df['p_val'].apply(format_pvalue)
    df['p_val_fdr_fmt'] = df['p_val_fdr_bh'].apply(format_pvalue)
    
    # Format mean difference
    df['mean_diff_fmt'] = df['mean_diff'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '--')
    
    # Format rank as integer
    df['rank_fmt'] = df['p_val_rank'].apply(lambda x: f'{int(x)}' if pd.notna(x) else '--')
    
    # Select columns for output
    output_df = df[['analysis_label', 'model_short', 'benchmark_short', 
                    'mean_diff_fmt', 'p_val_fmt', 'rank_fmt', 'p_val_fdr_fmt', 'n_samples']]
    output_df.columns = ['Analysis Set', 'Model', 'Benchmark', 'Mean Diff', 
                        'p-value', 'Rank', 'FDR-corrected p', 'n']
    
    # Sort by analysis set, model, and rank
    output_df = output_df.sort_values(['Analysis Set', 'Model', 'Rank'])
    
    # Create LaTeX table
    latex_table = r"""\footnotesize
\begin{longtable}{llllcccc}
\caption{Scalp EEG Model Statistical Comparisons} \label{tab:scalp_stats} \\
\toprule
\textbf{Analysis Set} & \textbf{Model} & \textbf{Benchmark} & \textbf{Mean Diff} & \textbf{p-value} & \textbf{Rank} & \textbf{FDR-corrected p} & \textbf{n} \\
\midrule
\endfirsthead

\multicolumn{8}{c}%
{{\tablename\ \thetable{} -- continued from previous page}} \\
\toprule
\textbf{Analysis Set} & \textbf{Model} & \textbf{Benchmark} & \textbf{Mean Diff} & \textbf{p-value} & \textbf{Rank} & \textbf{FDR-corrected p} & \textbf{n} \\
\midrule
\endhead

\midrule \multicolumn{8}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

"""
    
    # Generate table rows
    table_rows = []
    for idx, row in output_df.iterrows():
        row_str = f"{row['Analysis Set']} & {row['Model']} & {row['Benchmark']} & {row['Mean Diff']} & {row['p-value']} & {row['Rank']} & {row['FDR-corrected p']} & {int(row['n'])} \\\\"
        table_rows.append(row_str)
    
    table_body = '\n'.join(table_rows)
    full_table = latex_table + table_body + "\n\\end{longtable}\n"
    
    # Output to file
    output_file = ospj(prodatapath,'table_scalp_stats.tex')
    with open(output_file, 'w') as f:
        f.write(full_table)
    
    print(f"✓ Scalp statistics table generated: {output_file}")
    print(f"  Total rows: {len(output_df)}")
    print(f"\nTo use in your manuscript:")
    print(f"  \\input{{{output_file}}}")

if __name__ == "__main__":
    main()
