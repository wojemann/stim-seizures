#!/usr/bin/env python3
"""
Generate table-only LaTeX code for scalp EEG model performance.
Shows AUROC metrics (mean and median with confidence intervals) for different models and analysis sets.
"""

import pandas as pd
import numpy as np
from config import Config
from os.path import join as ospj

prodatapath = Config.deal(['prodatapath'])

def main():
    # Read the CSV
    df = pd.read_csv(ospj(prodatapath,'scalp_model_performance_table.csv'))
    
    # Custom model name mapping (shorten the long names)
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
        'old': 'Validation Set',
        'new': 'Test Set',
        'old+new': 'Combined'
    }
    
    # Apply mappings
    df['model_short'] = df['model'].map(model_names).fillna(df['model'])
    df['analysis_label'] = df['analysis_set'].map(analysis_labels).fillna(df['analysis_set'])
    
    # Format mean AUROC with CI
    def format_mean_auroc(row):
        mean = row['AUROC_mean']
        lower = row['AUROC_mean_ci_l']
        upper = row['AUROC_mean_ci_u']
        
        if pd.isna(mean):
            return '--'
        
        # Calculate half CI width
        ci_half = (upper - lower) / 2.0
        return f"{mean:.2f} $\\pm$ {ci_half:.2f}"
    
    # Format median AUROC with CI
    def format_median_auroc(row):
        median = row['AUROC_median']
        lower = row['AUROC_median_ci_l']
        upper = row['AUROC_median_ci_u']
        
        if pd.isna(median):
            return '--'
        
        # Calculate half CI width
        ci_half = (upper - lower) / 2.0
        return f"{median:.2f} $\\pm$ {ci_half:.2f}"
    
    df['mean_formatted'] = df.apply(format_mean_auroc, axis=1)
    df['median_formatted'] = df.apply(format_median_auroc, axis=1)
    
    # Select columns for output
    output_df = df[['analysis_label', 'model_short', 'mean_formatted', 'median_formatted', 'AUROC_n']]
    output_df.columns = ['Analysis Set', 'Model', 'Mean AUROC', 'Median AUROC', 'n']
    
    # Sort by analysis set and model
    sort_order = {'Validation Set': 0, 'Test Set': 1, 'Combined': 2}
    output_df['sort_key'] = output_df['Analysis Set'].map(sort_order)
    output_df = output_df.sort_values(['sort_key', 'Model']).drop('sort_key', axis=1)
    
    # Create LaTeX table
    latex_table = r"""\footnotesize
\begin{longtable}{llccc}
\caption{Scalp EEG Model Performance: AUROC Metrics} \label{tab:scalp_performance} \\
\toprule
\textbf{Analysis Set} & \textbf{Model} & \textbf{Mean AUROC} & \textbf{Median AUROC} & \textbf{n} \\
\midrule
\endfirsthead

\multicolumn{5}{c}%
{{\tablename\ \thetable{} -- continued from previous page}} \\
\toprule
\textbf{Analysis Set} & \textbf{Model} & \textbf{Mean AUROC} & \textbf{Median AUROC} & \textbf{n} \\
\midrule
\endhead

\midrule \multicolumn{5}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

"""
    
    # Generate table rows
    table_rows = []
    for idx, row in output_df.iterrows():
        row_str = f"{row['Analysis Set']} & {row['Model']} & {row['Mean AUROC']} & {row['Median AUROC']} & {int(row['n'])} \\\\"
        table_rows.append(row_str)
    
    table_body = '\n'.join(table_rows)
    full_table = latex_table + table_body + "\n\\end{longtable}\n"
    
    # Output to file
    output_file = ospj(prodatapath,'table_scalp_performance.tex')
    with open(output_file, 'w') as f:
        f.write(full_table)
    
    print(f"✓ Scalp performance table generated: {output_file}")
    print(f"  Total rows: {len(output_df)}")
    print(f"\nTo use in your manuscript:")
    print(f"  \\input{{{output_file}}}")

if __name__ == "__main__":
    main()
