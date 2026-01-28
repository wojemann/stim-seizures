import pandas as pd
import numpy as np
from config import Config
from os.path import join as ospj

prodatapath = Config.deal(['prodatapath'])
# Read the CSV
df = pd.read_csv(ospj(prodatapath,'model_performance_stats.csv'))

# Format numeric columns
def format_pvalue(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        v = float(val)
        if v < 0.001:
            return f'{v:.2e}'
        else:
            return f'{v:.3f}'
    except:
        return str(val)

def format_intercept(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        v = float(val)
        return f'{v:.2f}'
    except:
        return str(val)

# Format rank as integer
def format_rank(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        return f'{int(float(val))}'
    except:
        return str(val)

# Apply formatting
df['p_value_fmt'] = df['p_value'].apply(format_pvalue)
df['intercept_fmt'] = df['intercept'].apply(format_intercept)
df['p_rank_fmt'] = df['p_rank'].apply(format_rank)
df['p_fdr_corrected_fmt'] = df['p_fdr_corrected'].apply(format_pvalue)

# Select columns for output
output_df = df[['stage', 'model1', 'model2', 'metric', 
                'p_value_fmt', 'intercept_fmt', 'p_rank_fmt', 'p_fdr_corrected_fmt']]

# Start building the LaTeX document
latex_preamble = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage[margin=0.5in]{geometry}

\begin{document}

"""

latex_table_start = r"""\small
\begin{longtable}{llllcccc}
\caption{Model Performance Comparison Statistics} \label{tab:model_performance} \\
\toprule
\textbf{Stage} & \textbf{Model 1} & \textbf{Model 2} & \textbf{Metric} & \textbf{p-value} & \textbf{Intercept} & \textbf{Rank} & \textbf{FDR-corrected p} \\
\midrule
\endfirsthead

\multicolumn{8}{c}%
{{\tablename\ \thetable{} -- continued from previous page}} \\
\toprule
\textbf{Stage} & \textbf{Model 1} & \textbf{Model 2} & \textbf{Metric} & \textbf{p-value} & \textbf{Intercept} & \textbf{Rank} & \textbf{FDR-corrected p} \\
\midrule
\endhead

\midrule \multicolumn{8}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

"""

latex_table_end = r"""
\end{longtable}
\end{document}
"""

# Generate table body
table_rows = []
for idx, row in output_df.iterrows():
    # Escape underscores in metric names
    metric = row['metric'].replace('_', r'\_')
    row_str = f"{row['stage']} & {row['model1']} & {row['model2']} & {metric} & {row['p_value_fmt']} & {row['intercept_fmt']} & {row['p_rank_fmt']} & {row['p_fdr_corrected_fmt']} \\\\"
    table_rows.append(row_str)

table_body = '\n'.join(table_rows)

# Combine everything
full_latex = latex_preamble + latex_table_start + table_body + latex_table_end

# Save to file
with open(ospj(prodatapath,'model_performance_complete.tex'), 'w') as f:
    f.write(full_latex)

print("Complete LaTeX document generated successfully!")
print(f"Total rows: {len(df)}")
print(f"\nTo compile: pdflatex {ospj(prodatapath,'model_performance_complete.tex')}")
