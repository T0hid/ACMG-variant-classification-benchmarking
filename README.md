ACMG Variant Classification Benchmarking
A Python tool for benchmarking various variant classification tools (Franklin, GeneBE, InterVar, TAPES, LIRICAL, CPSR) against manual annotations.
Overview
This tool evaluates the performance of different variant classification tools by comparing their results with expert manual annotations. It generates publication-ready figures and comprehensive statistical reports.
Features

Multi-tool comparison: Benchmarks Franklin, GeneBE, InterVar, TAPES, LIRICAL, and CPSR
Separate analysis: Handles Mendelian and cancer samples separately
Comprehensive metrics:

Ranking accuracy (Top-1, Top-5, Top-10, etc.)
Retention rates
F1 scores and recall
AUC analysis
Statistical validation (Friedman test, Nemenyi post-hoc, Fisher's exact test)


Publication-ready figures: Generates multi-panel figures suitable for scientific publications

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

ACMG_185.csv: Unified ACMG data containing variant classifications from all tools
LIRICAL_185.tsv: LIRICAL ranking data
hgnc_standardized_matched_manual_annotations_185.xlsx: Manual annotations (ground truth)


Update file paths in the script:
pythonbase_dir = "path/to/your/data"
unified_acmg_path = os.path.join(base_dir, "ACMG_185.csv")
lirical_folder_path = base_dir
manual_annotations_path = os.path.join(base_dir, "manual_annotations.xlsx")

Run the benchmarking:
bashpython variant_benchmarking.py
