Comprehensive Evaluation of ACMG/AMP-based Variant Classification Tools 
 
Overview


Python tool for benchmarking various variant classification tools (Franklin, GeneBE, InterVar, TAPES, LIRICAL) against manual annotations.
This tool evaluates the performance of different variant classification tools by comparing their results with expert manual annotations. 


Features


Multi-tool comparison: Benchmarks Franklin, GeneBE, InterVar, TAPES(ACMG/AMP-based) LIRICAL(Top-ranked phenotype driven tools)


Comprehensive metrics:
Ranking accuracy (Top-1, Top-5, Top-10, etc.)
Retention rates
F1 scores and recall
AUC analysis
Empirical cumulative distribution function (CDF)
Statistical validation (Friedman test,confidence interval)


Requirements
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
statsmodels
lifelines
scikit-posthocs
matplotlib-venn
Installation

Clone this repository:
bashgit clone https://github.com/T0hid/ACMG-variant-classification-benchmarking.git
cd ACMG-variant-classification-benchmarking

Install dependencies:
bashpip install -r requirements.txt


Usage

Download  the  data files :

ACMG_163new.CSV
Unified ACMG data containing variant classifications from all tools     
                                               
LIRICAL_163new.CSV
LIRICAL ranking data

manual_163new.xlsx: 
Manual annotations (ground truth)


Update file paths in the script:
pythonbase_dir = "path/to/your/data"
Run the benchmarking:
bashpython variant_benchmarking.py

Contact information:
tohid.ghasemnejad@unsw.edu.au
