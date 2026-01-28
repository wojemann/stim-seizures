#!/usr/bin/env python3
"""
Generate table-only LaTeX code for combined model performance.
Pivots the data so each row is a metric+stage, and each column is a model.
Output format: mean ± 95% CI
"""

import pandas as pd
import numpy as np
import sys
from config import Config
from os.path import join as ospj

prodatapath = Config.deal(['prodatapath'])

def main():
    # Read the CSV
    df = pd.read_csv(ospj(prodatapath,'combined_model_performance.csv'))
    
    # Custom metric name mapping (based on your provided format)
    metric_names = {
        'max_onset_phi': 'opt. onset $\\phi$',
        'max_spread_phi': 'opt. spread $\\phi$',
        'onset_auc': 'onset auroc',
        'onset_auprc_normalized': 'onset auprc',
        'onset_phi_at_learned_f1_plateau_median': 'pre. onset $\\phi$',
        'spread_auc': 'spread auroc',
        'spread_auprc_normalized': 'spread auprc',
        'spread_phi_at_learned_f1_plateau_median': 'pre. spread $\\phi$',
        'adj_med_soz_spread_rank_pct': 'spread rank',
        'auc': 'auc',
        'auprc_raw': 'auprc',
        'avg_soz_recruitment_latency': 'recruitment latency',
        'f1': 'f1',
        'phi': '$\\phi$',
        'precision': 'precision',
        'sensitivity': 'sensitivity',
        'specificity': 'specificity'
    }
    
    # Create formatted string: mean ± half_CI_width (2 decimals)
    def format_mean_ci(row):
        if pd.isna(row['mean']) or row['mean'] == '':
            return '--'
        
        mean = float(row['mean'])
        lower = float(row['lower_ci'])
        upper = float(row['upper_ci'])
        
        # Calculate half the CI width (this is 1.96 * SE for 95% CI)
        ci_width = upper - lower
        half_ci = ci_width / 2.0
        
        # Format as: mean ± half_CI (2 decimals)
        return f"{mean:.2f} $\\pm$ {half_ci:.2f}"
    
    df['formatted'] = df.apply(format_mean_ci, axis=1)
    
    # Create a unique identifier for each metric+stage combination
    df['metric_stage'] = df['stage'] + '_' + df['metric']
    
    # Pivot: rows = metric_stage, columns = model
    pivot_df = df.pivot(index='metric_stage', columns='model', values='formatted')
    
    # Split the metric_stage back into stage and metric for better organization
    pivot_df['stage'] = pivot_df.index.str.split('_').str[0]
    pivot_df['metric'] = pivot_df.index.str.split('_', n=1).str[1]
    
    # Reorder columns: stage, metric, then all models
    # Get unique models (columns that are not stage/metric)
    model_columns = [col for col in pivot_df.columns if col not in ['stage', 'metric']]
    
    # Reorder
    pivot_df = pivot_df[['stage', 'metric'] + model_columns]
    
    # Sort by stage and metric
    pivot_df = pivot_df.sort_values(['stage', 'metric'])
    
    # Replace NaN with '--'
    pivot_df = pivot_df.fillna('--')
    
    # Apply custom metric names
    pivot_df['metric'] = pivot_df['metric'].map(metric_names).fillna(pivot_df['metric'])
    
    # Create LaTeX table
    num_models = len(model_columns)
    
    # Build column format string (ll for stage/metric, then c for each model)
    col_format = 'll' + 'c' * num_models
    
    # Build header
    header_cols = ['\\textbf{Stage}', '\\textbf{Metric}'] + [f'\\textbf{{{m}}}' for m in model_columns]
    header = ' & '.join(header_cols) + ' \\\\'
    
    # Start building the table
    latex_table = f"""\\footnotesize
\\begin{{longtable}}{{{col_format}}}
\\caption{{Model Performance Summary: Mean $\\pm$ 95\\% CI}} \\label{{tab:model_performance_summary}} \\\\
\\toprule
{header}
\\midrule
\\endfirsthead

\\multicolumn{{{num_models + 2}}}{{c}}%
{{{{\\tablename\\ \\thetable{{}} -- continued from previous page}}}} \\\\
\\toprule
{header}
\\midrule
\\endhead

\\midrule \\multicolumn{{{num_models + 2}}}{{r}}{{{{Continued on next page}}}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

"""
    
    # Generate table rows
    table_rows = []
    for idx, row in pivot_df.iterrows():
        # Create row string
        row_values = [str(row['stage']), str(row['metric'])] + [str(row[m]) for m in model_columns]
        row_str = ' & '.join(row_values) + ' \\\\'
        table_rows.append(row_str)
    
    table_body = '\n'.join(table_rows)
    full_table = latex_table + table_body + "\n\\end{longtable}\n"
    
    # Output to file
    output_file = ospj(prodatapath,'table_combined_performance.tex')
    with open(output_file, 'w') as f:
        f.write(full_table)
    
    print(f"✓ Pivoted table-only LaTeX generated: {output_file}")
    print(f"  Rows (metric-stage combinations): {len(pivot_df)}")
    print(f"  Columns (models): {model_columns}")
    print(f"\nTo use in your manuscript:")
    print(f"  \\input{{{output_file}}}")
    print(f"\nNote: Requires \\usepackage{{booktabs}} and \\usepackage{{longtable}} in preamble")

if __name__ == "__main__":
    main()
