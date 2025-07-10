Evaluation of ACMG/AMP-based Variant Classification Tools Highlights the Critical Importance of Phenotypic Data Incorporation.

 
Overview


Python tool for benchmarking various variant classification tools (Franklin, GeneBE, InterVar, TAPES, LIRICAL, CPSR) against manual annotations.
This tool evaluates the performance of different variant classification tools by comparing their results with expert manual annotations. 


Features


Multi-tool comparison: Benchmarks Franklin, GeneBE, InterVar, TAPES, LIRICAL, and CPSR


Separate analysis: Handles Mendelian and cancer samples separately
Comprehensive metrics:

Ranking accuracy (Top-1, Top-5, Top-10, etc.)
Retention rates
F1 scores and recall
AUC analysis
Statistical validation (Friedman test, Nemenyi post-hoc,)



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

Prepare your data files:

ACMG_data.csv:
Unified ACMG data containing variant classifications from all tools     
                                               
LIRICAL_data.tsv: 
LIRICAL ranking data

hgnc_standardized_manual_annotations_.xlsx: 
Manual annotations (ground truth)


Update file paths in the script:
pythonbase_dir = "path/to/your/data"
Run the benchmarking:
bashpython variant_benchmarking.py
