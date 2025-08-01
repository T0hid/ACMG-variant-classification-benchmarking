#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Variant Classification Tools Benchmarking Script with Journal-Compliant Figures
and a Comprehensive Text Report.

This script performs a unified analysis on a dataset, generates publication-ready
figures, and produces a detailed text report with all key performance metrics.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
from statsmodels.stats import inter_rater as irr
from statsmodels.stats.contingency_tables import mcnemar, SquareTable
import warnings
from collections import defaultdict
import itertools
from matplotlib_venn import venn2, venn3
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import statsmodels.api as sm
from sklearn.utils import resample
from scipy.stats import friedmanchisquare, fisher_exact
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import matplotlib.patches as patches
from statsmodels.stats.multitest import multipletests
import string

# JOURNAL-COMPLIANT SETTINGS
# Convert mm to inches for matplotlib (1 inch = 25.4 mm)
COLUMN_WIDTH_INCHES = 180 / 25.4  # 7.09 inches for 2-column width
SINGLE_COLUMN_WIDTH = 88 / 25.4   # 3.46 inches for 1-column width

# Set journal-compliant font sizes and style
plt.rcParams.update({
    'font.size': 6,          # Base font size (within 5-7pt range)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 6,         # Axis labels
    'axes.titlesize': 7,         # Subplot titles
    'xtick.labelsize': 5,        # X-tick labels (minimum allowed)
    'ytick.labelsize': 5,        # Y-tick labels (minimum allowed)
    'legend.fontsize': 5,        # Legend text (minimum allowed)
    'figure.titlesize': 7,       # Figure title (maximum allowed)
    'axes.linewidth': 0.5,       # Thinner lines for cleaner look
    'lines.linewidth': 1,        # Line plot width
    'lines.markersize': 3,       # Smaller markers
    'legend.frameon': True,      # Always show legend frame
    'legend.fancybox': False,    # Simple box
    'legend.shadow': False,      # No shadow for cleaner look
    'legend.borderpad': 0.3,     # Reduce padding
    'legend.columnspacing': 0.5, # Reduce column spacing
    'legend.handlelength': 1.0,  # Shorter legend handles
    'legend.handletextpad': 0.3, # Less space between handle and text
    'legend.borderaxespad': 0.3, # Less space to axes
    'figure.dpi': 1200,          # High resolution
    'savefig.dpi': 1200,         # High resolution for saving
    'savefig.bbox': 'tight',     # Tight bounding box
    'savefig.pad_inches': 0.05,  # Minimal padding
})

# Use paper context but override with our specific settings
sns.set_style("ticks")  # Clean style with ticks
sns.set_context("paper", rc={"lines.linewidth": 1})

# Suppress warnings
warnings.filterwarnings('ignore')

# Panel label styling function
def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add a panel label (A, B, C, etc.) to an axis with journal-compliant styling."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=7, fontweight='bold', va='bottom', ha='right')

class VariantToolBenchmarker:
    """Class to benchmark variant classification tools with journal-compliant visualization."""

    def __init__(self, acmg_path, lirical_folder, manual_path, exclude_tools=None, sample_subset=None, analysis_type='all'):
        """
        Initialize the benchmarker with paths to data files.

        Parameters:
        -----------
        acmg_path : str
            Path to unified ACMG data CSV file
        lirical_folder : str
            Path to folder containing LIRICAL files
        manual_path : str
            Path to manual annotations Excel file
        exclude_tools : list, optional
            List of tools to exclude from analysis (default: ['charger', 'cpsr'])
        sample_subset : list, optional
            List of sample IDs to include in the analysis (default: all samples)
        analysis_type : str
            A label for the analysis type (e.g., 'all', 'cancer', 'mendelian')
        """
        self.acmg_path = acmg_path
        self.lirical_folder = lirical_folder
        self.manual_path = manual_path
        self.exclude_tools = exclude_tools if exclude_tools is not None else ['charger', 'cpsr']
        self.sample_subset = sample_subset
        self.analysis_type = analysis_type

        # Initialize result containers
        self.results = {}
        self.tool_data = {}
        self.metrics = {}
        self.statistical_tests = {}  # To store statistical test results
        self.max_rank_length = 0 # For consistent AUC scoring

    def load_data(self):
        """Load and preprocess all data sources."""
        print(f"Loading and preprocessing data for '{self.analysis_type}' analysis...")

        # Load unified ACMG data
        self.acmg_data = pd.read_csv(self.acmg_path)
        print(f"Loaded ACMG data: {self.acmg_data.shape[0]} rows")

        # Filter out excluded tools
        self.acmg_filtered = self.acmg_data[~self.acmg_data['tool'].str.lower().isin([t.lower() for t in self.exclude_tools])]
        print(f"After filtering {', '.join(self.exclude_tools)}: {self.acmg_filtered.shape[0]} rows")

        # Load manual annotations (ground truth)
        self.manual_data = pd.read_excel(self.manual_path)
        print(f"Loaded manual annotations: {self.manual_data.shape[0]} rows")
        
        # Standardize sample ID column name
        sample_id_col = self.find_sample_id_column(self.manual_data)
        if sample_id_col and sample_id_col != 'Sample id':
            self.manual_data = self.manual_data.rename(columns={sample_id_col: 'Sample id'})
            print(f"Renamed '{sample_id_col}' column to 'Sample id'")

        # Filter by sample subset if provided
        if self.sample_subset:
            self.manual_data = self.manual_data[self.manual_data['Sample id'].isin(self.sample_subset)]
            self.acmg_filtered = self.acmg_filtered[self.acmg_filtered['sample_id'].isin(self.sample_subset)]
            print(f"Filtered to {len(self.sample_subset)} samples: {self.manual_data.shape[0]} manual annotations, {self.acmg_filtered.shape[0]} ACMG variants")

        # Load LIRICAL data with proper encoding
        self.lirical_data = self._load_lirical_data()
        if self.lirical_data is not None:
            print(f"Loaded LIRICAL data: {self.lirical_data.shape[0]} rows")
            # Filter by sample subset if provided
            if self.sample_subset:
                self.lirical_data = self.lirical_data[self.lirical_data['sample_id'].isin(self.sample_subset)]
                print(f"Filtered LIRICAL data to {len(self.sample_subset)} samples: {self.lirical_data.shape[0]} rows")

        # Create a unified dataset by tool
        self._prepare_tool_data()
        print("Data loading and preprocessing complete.")

    def find_sample_id_column(self, df):
        """Finds the sample ID column from a list of possible names."""
        possible_names = ['Sample id', 'sample_id', 'Sample_id', 'SampleID', 'Sample ID', 'ID', 'sample']
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def _load_lirical_data(self):
        """Load LIRICAL data from a single combined TSV file."""
        lirical_file = os.path.join(self.lirical_folder, "LIRICAL_163new.tsv")

        if not os.path.exists(lirical_file):
            print(f"LIRICAL file not found: {lirical_file}")
            return None

        print(f"Loading combined LIRICAL file: {lirical_file}")

        try:
            # Load TSV file
            lirical_data = pd.read_csv(lirical_file, sep='\t')

            # Verify required columns exist
            required_columns = ['rank', 'sample_id', 'hgnc_gene']
            missing_columns = [col for col in required_columns if col not in lirical_data.columns]

            if missing_columns:
                print(f"Error: Missing required columns in LIRICAL data: {', '.join(missing_columns)}")
                return None

            # Ensure LIRICAL tool name is consistent
            lirical_data['tool'] = 'lirical'
            
            print(f"Loaded LIRICAL data with {len(lirical_data)} rows across {lirical_data['sample_id'].nunique()} samples")
            return lirical_data

        except Exception as e:
            print(f"Error loading LIRICAL data: {e}")
            return None

    def _prepare_tool_data(self):
        """Prepare data for each tool using native ranking systems."""
        tools = self.acmg_filtered['tool'].unique().tolist()
        
        if self.lirical_data is not None:
            tools.append('LIRICAL')
        
        # Ensure no excluded tools slip through
        tools = [t for t in tools if t.lower() not in [ex.lower() for ex in self.exclude_tools]]
        
        print(f"Preparing data for tools: {', '.join(tools)}")

        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}

        for tool in tools:
            tool_lower = tool.lower() # Case-insensitive comparison
            if tool_lower != 'lirical':
                tool_df = self.acmg_filtered[self.acmg_filtered['tool'].str.lower() == tool_lower].copy()
                
                if tool_df.empty:
                    print(f"Warning: No data found for tool {tool}")
                    continue
                
                print(f"Processing {tool} with {len(tool_df)} variants across {tool_df['sample_id'].nunique()} samples")
                
                sample_gene_rankings = {}
                for sample_id, group in tool_df.groupby('sample_id'):
                    if sample_id not in ground_truth:
                        continue
                    
                    # Preserve original order (native ranking) and remove duplicates
                    seen = set()
                    sorted_genes = [gene for gene in group['hgnc_gene'].tolist() if not (gene in seen or seen.add(gene))]
                    sample_gene_rankings[sample_id] = sorted_genes
                
                self.tool_data[tool] = sample_gene_rankings
                self._calculate_tool_summary(tool, sample_gene_rankings, ground_truth)

            else:  # Special handling for LIRICAL
                if self.lirical_data is not None:
                    print(f"Processing LIRICAL data with {len(self.lirical_data)} rows across {self.lirical_data['sample_id'].nunique()} samples")
                    
                    sample_gene_rankings = {}
                    for sample_id, group in self.lirical_data.groupby('sample_id'):
                        try:
                            sorted_genes_with_dupes = group.sort_values(by=['rank'], ascending=True)['hgnc_gene'].tolist()
                            seen = set()
                            sorted_genes = [g for g in sorted_genes_with_dupes if not (g in seen or seen.add(g))]
                            sample_gene_rankings[sample_id] = sorted_genes
                        except Exception as e:
                            print(f"Error processing LIRICAL data for sample {sample_id}: {e}")
                    
                    self.tool_data[tool] = sample_gene_rankings
                    self._calculate_tool_summary(tool, sample_gene_rankings, ground_truth)

    def _calculate_tool_summary(self, tool, sample_gene_rankings, ground_truth):
        """Calculate and display summary metrics for a tool."""
        found_count, top1_count, top5_count, top10_count = 0, 0, 0, 0
        
        # Only consider samples that the tool processed
        processed_samples = [s for s in ground_truth.keys() if s in sample_gene_rankings]
        total_samples = len(processed_samples)
        
        for sample_id in processed_samples:
            true_gene = ground_truth[sample_id]
            ranked_list = sample_gene_rankings[sample_id]
            if true_gene in ranked_list:
                found_count += 1
                rank = ranked_list.index(true_gene) + 1
                if rank == 1: top1_count += 1
                if rank <= 5: top5_count += 1
                if rank <= 10: top10_count += 1
        
        print(f"Tool {tool} summary:")
        print(f"  Processed {total_samples} out of {len(ground_truth)} ground truth samples.")
        if total_samples > 0:
            print(f"  Top-1 accuracy: {top1_count/total_samples*100:.1f}%")
            print(f"  Top-5 accuracy: {top5_count/total_samples*100:.1f}%")
            print(f"  Top-10 accuracy: {top10_count/total_samples*100:.1f}%")
            print(f"  Retention Rate (Found/Processed): {found_count/total_samples*100:.1f}%")

    def calculate_metrics(self):
        """Calculate all performance metrics for each tool."""
        print("Calculating performance metrics...")
        
        # Pre-calculate the max rank length across all tools for consistent AUC scoring
        max_length = 0
        for tool in self.tool_data.keys():
            tool_rankings = self.tool_data[tool]
            if not tool_rankings: continue
            max_len_tool = max((len(v) for v in tool_rankings.values()), default=0)
            if max_len_tool > max_length:
                max_length = max_len_tool
        self.max_rank_length = max_length

        for tool, rankings in self.tool_data.items():
            print(f"Processing tool: {tool}")
            
            ranking_metrics = self._calculate_ranking_metrics(rankings)
            filtering_metrics = self._calculate_filtering_metrics(rankings)
            accuracy_metrics = self._calculate_accuracy_metrics(rankings, ranking_metrics)
            distribution_metrics = self._calculate_distribution_metrics(rankings)

            self.metrics[tool] = {
                'ranking': ranking_metrics,
                'filtering': filtering_metrics,
                'accuracy': accuracy_metrics,
                'distribution': distribution_metrics
            }
        
        print("Performance metrics calculation complete.")

    def _calculate_ranking_metrics(self, rankings):
        """Calculate ranking metrics (Top-N accuracy, rank stats)."""
        metrics_rank = {}
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        found_ranks = []
        top_n_counts = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
        total_samples = 0
        
        # Iterate over all ground truth samples to correctly calculate accuracy
        for sample_id, true_gene in ground_truth.items():
            total_samples += 1
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                    found_ranks.append(rank)
                    for n in top_n_counts.keys():
                        if rank <= n:
                            top_n_counts[n] += 1
        
        if total_samples > 0:
            metrics_rank['top_rank_percentage'] = (top_n_counts[1] / total_samples) * 100
            for n, count in top_n_counts.items():
                metrics_rank[f'top_{n}_percentage'] = (count / total_samples) * 100
        else:
            metrics_rank['top_rank_percentage'] = np.nan
            for n in top_n_counts.keys():
                metrics_rank[f'top_{n}_percentage'] = np.nan
        
        if found_ranks:
            metrics_rank['mean_rank'] = np.mean(found_ranks)
            metrics_rank['median_rank'] = np.median(found_ranks)
        else:
            metrics_rank['mean_rank'] = np.nan
            metrics_rank['median_rank'] = np.nan
            
        return metrics_rank

    def _calculate_filtering_metrics(self, rankings):
        """Calculate filtering metrics (retention rate)."""
        metrics_filter = {}
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        retained_count, total_samples = 0, 0
        
        # Retention is based on samples the tool processed
        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                total_samples += 1
                if true_gene in rankings[sample_id]:
                    retained_count += 1
        
        if total_samples > 0:
            metrics_filter['retention_rate'] = (retained_count / total_samples) * 100
        else:
            metrics_filter['retention_rate'] = np.nan
            
        return metrics_filter

    def _calculate_accuracy_metrics(self, rankings, ranking_metrics):
        """Calculate accuracy metrics like Precision, Recall, F1, and AUC."""
        metrics_acc = {}
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        rank_thresholds = [1, 5, 10, 20, 50]
        
        # Mean Precision@k
        for k in rank_thresholds:
            precisions = []
            for sample_id, true_gene in ground_truth.items():
                if sample_id in rankings:
                    top_k_genes = rankings[sample_id][:k]
                    if not top_k_genes:
                        precisions.append(0.0)
                        continue
                    num_relevant_in_k = 1 if true_gene in top_k_genes else 0
                    precisions.append(num_relevant_in_k / k)
                else:
                    precisions.append(0.0)
            metrics_acc[f'precision_at_{k}'] = np.mean(precisions) if precisions else 0.0

        # Recall (based on all ground truth samples)
        true_positives = sum(1 for sid, tg in ground_truth.items() if sid in rankings and tg in rankings[sid])
        total_gt = len(ground_truth)
        recall = true_positives / total_gt if total_gt > 0 else np.nan
        metrics_acc['recall'] = recall

        # F1 Score (based on Top-1 Accuracy and overall recall)
        precision_at_1_by_sample = ranking_metrics.get('top_rank_percentage', 0) / 100
        if not np.isnan(recall) and precision_at_1_by_sample + recall > 0:
            metrics_acc['f1_score'] = 2 * (precision_at_1_by_sample * recall) / (precision_at_1_by_sample + recall)
        else:
            metrics_acc['f1_score'] = np.nan
        
        # ROC/AUC Calculation
        metrics_acc['auc'] = np.nan
        all_scores, all_labels = [], []
        max_length = self.max_rank_length

        if max_length > 0:
            for sample_id, true_gene in ground_truth.items():
                if sample_id in rankings:
                    ranked_genes = rankings[sample_id]
                    for i, gene in enumerate(ranked_genes):
                        all_scores.append(max_length - i)
                        all_labels.append(1 if gene == true_gene else 0)
                    if true_gene not in ranked_genes:
                        all_scores.append(0)
                        all_labels.append(1)
                else:
                    all_scores.append(0)
                    all_labels.append(1)

            if len(all_labels) > 0 and len(set(all_labels)) > 1:
                fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
                metrics_acc['auc'] = metrics.auc(fpr, tpr)
                
        return metrics_acc

    def _calculate_distribution_metrics(self, rankings):
        """Calculate rank distribution and data for ECDF."""
        metrics_dist = {}
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        categories = ['1st', '2-5', '6-10', '>10', 'Not Found']
        counts = {cat: 0 for cat in categories}
        total_samples = len(ground_truth)
        ecdf_ranks = []

        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                    ecdf_ranks.append(rank)
                    if rank == 1: counts['1st'] += 1
                    elif 2 <= rank <= 5: counts['2-5'] += 1
                    elif 6 <= rank <= 10: counts['6-10'] += 1
                    else: counts['>10'] += 1
                else:
                    counts['Not Found'] += 1
                    ecdf_ranks.append(len(ranked_genes) + 1) 
            else:
                counts['Not Found'] += 1
                ecdf_ranks.append(total_samples + 1) 

        metrics_dist['rank_distribution_counts'] = counts
        metrics_dist['rank_distribution_percentage'] = {k: (v / total_samples * 100) if total_samples > 0 else 0 for k, v in counts.items()}
        metrics_dist['ecdf_ranks'] = ecdf_ranks
        
        return metrics_dist

    def perform_statistical_tests(self):
        """Perform statistical tests (Bootstrap and Friedman) and store results."""
        print(f"Performing statistical tests for {self.analysis_type} analysis...")
        
        self.statistical_tests = {'bootstrap_ci': {}, 'friedman_test': {}}
        tools = list(self.tool_data.keys())
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        sample_ids = sorted(list(ground_truth.keys()))
        n_samples = len(sample_ids)

        # --- Bootstrap Confidence Intervals ---
        n_resamples = 1000
        metrics_list = ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']
        
        for tool in tools:
            self.statistical_tests['bootstrap_ci'][tool] = {}
            for metric in metrics_list:
                threshold = 1 if metric == 'top_rank_percentage' else int(metric.split('_')[1])
                samples_binary = []
                for sample_id in sample_ids:
                    found = False
                    if sample_id in self.tool_data[tool]:
                        ranked_genes = self.tool_data[tool][sample_id]
                        if ground_truth[sample_id] in ranked_genes and ranked_genes.index(ground_truth[sample_id]) < threshold:
                            found = True
                    samples_binary.append(1 if found else 0)
                
                if len(samples_binary) < 5:
                    self.statistical_tests['bootstrap_ci'][tool][metric] = {'lower': np.nan, 'upper': np.nan, 'mean': np.nan}
                    continue
                
                bootstrap_results = [np.mean(resample(samples_binary)) * 100 for _ in range(n_resamples)]
                lower, upper = np.percentile(bootstrap_results, [2.5, 97.5])
                self.statistical_tests['bootstrap_ci'][tool][metric] = {'lower': lower, 'upper': upper, 'mean': np.mean(bootstrap_results)}

        # --- Friedman Test ---
        if n_samples >= 5 and len(tools) >= 2:
            rank_matrix = []
            max_rank_penalty = n_samples + 1 # Penalty for not processing a sample
            for sample_id in sample_ids:
                sample_ranks = []
                for tool in tools:
                    if sample_id in self.tool_data[tool]:
                        ranked_genes = self.tool_data[tool][sample_id]
                        rank = ranked_genes.index(ground_truth[sample_id]) + 1 if ground_truth[sample_id] in ranked_genes else len(ranked_genes) + 1
                    else:
                        rank = max_rank_penalty 
                    sample_ranks.append(rank)
                rank_matrix.append(sample_ranks)
            
            rank_matrix = np.array(rank_matrix)
            
            try:
                statistic, p_value = friedmanchisquare(*rank_matrix.T)
                self.statistical_tests['friedman_test'] = {'statistic': statistic, 'p_value': p_value, 'rank_matrix': rank_matrix}
                print(f"Friedman test: statistic={statistic:.2f}, p-value={p_value:.4f}")
            except Exception as e:
                print(f"Friedman test error: {e}")
                self.statistical_tests['friedman_test'] = {'p_value': np.nan, 'rank_matrix': None}
        else:
            print("Need >=5 samples and >=2 tools for Friedman test")
            self.statistical_tests['friedman_test'] = {'p_value': np.nan, 'rank_matrix': None}

    def generate_report(self, output_file='benchmark_report.txt'):
        """Generate a comprehensive report of the results."""
        print(f"Generating comprehensive report and saving to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"  VARIANT CLASSIFICATION BENCHMARKING REPORT ({self.analysis_type.upper()})\n")
            f.write("="*80 + "\n\n")
            
            sorted_tools = sorted(self.tool_data.keys(), key=lambda t: self.metrics[t]['ranking'].get('top_10_percentage', 0), reverse=True)
            f.write("TOOLS ANALYZED\n" + "-"*15 + "\n")
            for tool in sorted_tools:
                f.write(f"- {tool}\n")
            f.write("\n")
            
            # --- Main Performance Table ---
            f.write("OVERALL PERFORMANCE METRICS\n" + "-"*30 + "\n")
            headers = ['Tool', 'Top-1 %', 'Top-5 %', 'Top-10 %', 'Top-50 %', 'Retention %', 'Mean Rank', 'Median Rank', 'AUC']
            f.write(f"{headers[0]:<12}{headers[1]:<10}{headers[2]:<10}{headers[3]:<10}{headers[4]:<10}{headers[5]:<12}{headers[6]:<12}{headers[7]:<14}{headers[8]:<10}\n")
            f.write("-" * 100 + "\n")

            for tool in sorted_tools:
                ranking = self.metrics[tool]['ranking']
                filtering = self.metrics[tool]['filtering']
                accuracy = self.metrics[tool]['accuracy']
                
                top1 = ranking.get('top_rank_percentage', 0)
                top5 = ranking.get('top_5_percentage', 0)
                top10 = ranking.get('top_10_percentage', 0)
                top50 = ranking.get('top_50_percentage', 0)
                retention = filtering.get('retention_rate', 0)
                mean_r = ranking.get('mean_rank', np.nan)
                median_r = ranking.get('median_rank', np.nan)
                auc = accuracy.get('auc', np.nan)

                f.write(f"{tool:<12}{top1:<10.2f}{top5:<10.2f}{top10:<10.2f}{top50:<10.2f}{retention:<12.2f}{f'{mean_r:.2f}':<12}{f'{median_r:.2f}':<14}{f'{auc:.3f}':<10}\n")
            f.write("\n")

            # --- Rank Distribution Table ---
            f.write("RANK DISTRIBUTION (% of Cases)\n" + "-"*30 + "\n")
            dist_headers = ['Tool', 'Rank 1', 'Rank 2-5', 'Rank 6-10', 'Rank >10', 'Not Found']
            f.write(f"{dist_headers[0]:<12}{dist_headers[1]:<12}{dist_headers[2]:<12}{dist_headers[3]:<12}{dist_headers[4]:<12}{dist_headers[5]:<12}\n")
            f.write("-" * 72 + "\n")
            for tool in sorted_tools:
                dist_pct = self.metrics[tool]['distribution']['rank_distribution_percentage']
                f.write(f"{tool:<12}{dist_pct['1st']:<12.2f}{dist_pct['2-5']:<12.2f}{dist_pct['6-10']:<12.2f}{dist_pct['>10']:<12.2f}{dist_pct['Not Found']:<12.2f}\n")
            f.write("\n")

            # --- Cumulative Probability Table (ECDF data) ---
            f.write("CUMULATIVE PROBABILITY (ECDF) (% of Cases)\n" + "-"*45 + "\n")
            ecdf_headers = ['Tool', 'Rank <= 1', 'Rank <= 5', 'Rank <= 10', 'Rank <= 50']
            f.write(f"{ecdf_headers[0]:<12}{ecdf_headers[1]:<12}{ecdf_headers[2]:<12}{ecdf_headers[3]:<12}{ecdf_headers[4]:<12}\n")
            f.write("-" * 60 + "\n")
            for tool in sorted_tools:
                ranks = np.array(self.metrics[tool]['distribution'].get('ecdf_ranks', []))
                f.write(f"{tool:<12}")
                if len(ranks) > 0:
                    for threshold in [1, 5, 10, 50]:
                        prob = (np.sum(ranks <= threshold) / len(ranks)) * 100
                        f.write(f"{prob:<12.2f}")
                else:
                    f.write(f"{'N/A':<12}"*4)
                f.write("\n")
            f.write("\n")

            # --- Precision and F1 Score Table ---
            f.write("PRECISION & F1 SCORE\n" + "-"*30 + "\n")
            prec_headers = ['Tool', 'P@1', 'P@5', 'P@10', 'Recall', 'F1 Score']
            f.write(f"{prec_headers[0]:<12}{prec_headers[1]:<10}{prec_headers[2]:<10}{prec_headers[3]:<10}{prec_headers[4]:<10}{prec_headers[5]:<10}\n")
            f.write("-" * 62 + "\n")
            for tool in sorted_tools:
                accuracy = self.metrics[tool]['accuracy']
                p1 = accuracy.get('precision_at_1', np.nan)
                p5 = accuracy.get('precision_at_5', np.nan)
                p10 = accuracy.get('precision_at_10', np.nan)
                recall = accuracy.get('recall', np.nan)
                f1 = accuracy.get('f1_score', np.nan)
                f.write(f"{tool:<12}{p1:<10.3f}{p5:<10.3f}{p10:<10.3f}{recall:<10.3f}{f1:<10.3f}\n")
            f.write("\n")

            # Statistical Tests
            self.update_report_with_statistics(f)

    def update_report_with_statistics(self, f):
        """Update the report with statistical test results."""
        f.write("STATISTICAL TESTS\n" + "-"*20 + "\n")
        
        # Friedman Test
        if 'friedman_test' in self.statistical_tests and self.statistical_tests['friedman_test'].get('p_value') is not None:
            friedman = self.statistical_tests['friedman_test']
            if not np.isnan(friedman.get('p_value')):
                f.write("Friedman Test (Overall Ranking Performance):\n")
                f.write(f"  Statistic: {friedman.get('statistic', 'N/A'):.4f}\n")
                f.write(f"  p-value: {friedman.get('p_value', 'N/A'):.4f}\n")
                f.write(f"  Significant difference among tools: {'Yes' if friedman.get('p_value') < 0.05 else 'No'}\n\n")
            else:
                f.write("  Friedman test could not be performed (insufficient data).\n\n")

        # Bootstrap Confidence Intervals
        if 'bootstrap_ci' in self.statistical_tests:
            f.write("Bootstrap 95% Confidence Intervals (Top-N Accuracy):\n")
            f.write(f"{'Tool':<15}{'Top-1 (%)':<25}{'Top-5 (%)':<25}{'Top-10 (%)':<25}\n")
            f.write("-" * 90 + "\n")
            sorted_tools = sorted(self.tool_data.keys(), key=lambda t: self.metrics[t]['ranking'].get('top_10_percentage', 0), reverse=True)
            for tool in sorted_tools:
                if tool in self.statistical_tests['bootstrap_ci']:
                    ci_data = self.statistical_tests['bootstrap_ci'][tool]
                    f.write(f"{tool:<15}")
                    for metric in ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']:
                        if metric in ci_data and 'mean' in ci_data[metric]:
                            data = ci_data[metric]
                            f.write(f"{data['mean']:.2f} ({data['lower']:.2f}-{data['upper']:.2f}){' ':<8}")
                        else:
                            f.write(f"{'N/A':<25}")
                    f.write("\n")
            f.write("\n")


def generate_figures(benchmarker, output_dir='publication_figures'):
    """Generate the final set of journal-compliant figures for the unified analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Setup: Colors and Data ---
    tool_colors = {
        'franklin': '#1f77b4', 'Franklin': '#1f77b4',
        'genebe': '#ff7f0e',   'Genebe': '#ff7f0e',
        'intervar': '#2ca02c', 'Intervar': '#2ca02c', 'InterVar': '#2ca02c',
        'tapes': '#d62728',    'TAPES': '#d62728',
        'lirical': '#9467bd',  'LIRICAL': '#9467bd',
        'cpsr': '#8c564b',     'CPSR': '#8c564b'
    }
    ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in benchmarker.manual_data.iterrows()}
    
    all_tools = list(benchmarker.tool_data.keys())
    tool_order = ['Franklin', 'Genebe', 'Intervar', 'TAPES', 'LIRICAL']
    tool_order = [t for t in tool_order if t in all_tools]

    tool_labels = {
        t: t.upper() if t.lower() in ['lirical', 'tapes'] else t.capitalize() 
        for t in all_tools
    }

    # =============================================================================
    # Figure 1: Performance Overview
    # =============================================================================
    print("Generating Figure 1: Performance Overview...")
    fig = plt.figure(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES))
    
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.5) 
    axA = fig.add_subplot(gs[0, 0], polar=True)
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # --- Panel A: Radar plot (with Top-50) ---
    metrics_radar = ['Top-1 %', 'Top-10 %', 'Top-50 %', 'Recall (%)']
    N = len(metrics_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    axA.set_theta_offset(np.pi / 2)
    axA.set_theta_direction(-1)
    axA.set_xticks(angles[:-1])
    axA.set_xticklabels(metrics_radar, fontsize=5)
    axA.set_ylim(0, 100)
    for tool in tool_order:
        values = [
            benchmarker.metrics[tool]['ranking'].get('top_rank_percentage', 0),
            benchmarker.metrics[tool]['ranking'].get('top_10_percentage', 0),
            benchmarker.metrics[tool]['ranking'].get('top_50_percentage', 0),
            benchmarker.metrics[tool]['accuracy'].get('recall', 0) * 100
        ] + [benchmarker.metrics[tool]['ranking'].get('top_rank_percentage', 0)]
        axA.plot(angles, values, linewidth=1.2, linestyle='solid', label=tool_labels[tool], color=tool_colors.get(tool, None), marker='o', markersize=2, alpha=0.8)
        axA.fill(angles, values, alpha=0.2, color=tool_colors.get(tool, None))
    add_panel_label(axA, 'A')
    axA.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(tool_order), fontsize=5, frameon=True)

    # --- Panel B: Top-N Ranking Accuracy (with Top-50 and labels) ---
    thresholds = [1, 5, 10, 20, 50]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    accuracy_data = np.array([[benchmarker.metrics[tool]['ranking'].get(f'top_{t}_percentage', 0) for t in thresholds] for tool in tool_order])
    width = 0.15 
    x = np.arange(len(tool_order))
    for i, threshold in enumerate(thresholds):
        offset = width * (i - len(thresholds) / 2 + 0.5)
        bars = axB.bar(x + offset, accuracy_data[:, i], width, label=f'Top-{threshold}', color=colors[i])
        axB.bar_label(bars, fmt='%.0f%%', fontsize=4, padding=2, rotation=90)
    add_panel_label(axB, 'B')
    axB.set_ylabel('Accuracy (%)', fontsize=6)
    axB.set_xticks(x)
    axB.set_xticklabels([tool_labels[t] for t in tool_order], fontsize=5, rotation=45, ha='right', fontweight='bold')
    axB.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=5, frameon=True)
    axB.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    axB.set_ylim(0, 115) 
    
    # --- Panel C: Retention Rate (with labels) ---
    retention = [benchmarker.metrics[tool]['filtering'].get('retention_rate', 0) for tool in tool_order]
    bars = axC.bar(range(len(tool_order)), retention, color='#4CAF50')
    add_panel_label(axC, 'C')
    axC.set_ylabel('Retention Rate (%)', fontsize=6)
    axC.set_xticks(range(len(tool_order)))
    axC.set_xticklabels([tool_labels[t] for t in tool_order], fontsize=5, rotation=45, ha='right', fontweight='bold')
    axC.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    axC.set_ylim(0, 105)
    axC.bar_label(bars, fmt='%.0f%%', fontsize=5, padding=2)

    # --- Panel D: Rank Distribution (with labels) ---
    categories = ['1st', '2-5', '6-10', '>10', 'Not Found']
    colors_stacked = ['#4daf4a', '#ff7f00', '#377eb8', '#984ea3', '#e41a1c']
    rank_distributions = []
    for tool in tool_order:
        dist_pct = benchmarker.metrics[tool]['distribution']['rank_distribution_percentage']
        rank_distributions.append([dist_pct[cat] for cat in categories])
    
    data = np.array(rank_distributions).T
    bottoms = np.zeros(len(tool_order))
    for i, cat in enumerate(categories):
        bars = axD.bar(range(len(tool_order)), data[i], bottom=bottoms, label=cat, color=colors_stacked[i])
        for j, bar in enumerate(bars):
            val = data[i, j]
            if val > 6:
                axD.text(bar.get_x() + bar.get_width()/2, 
                         bottoms[j] + val/2, 
                         f'{val:.0f}%', 
                         ha='center', va='center', fontsize=4, color='black', fontweight='bold')
        bottoms += data[i]

    add_panel_label(axD, 'D')
    axD.set_ylabel('Percentage of Cases (%)', fontsize=6)
    axD.set_xticks(range(len(tool_order)))
    axD.set_xticklabels([tool_labels[t] for t in tool_order], fontsize=5, rotation=45, ha='right', fontweight='bold')
    
    handles, labels = axD.get_legend_handles_labels()
    axD.legend(handles[::-1], labels[::-1], title='Rank', loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=5)
    axD.set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Figure1_performance_overview.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'Figure1_performance_overview.png'), format='png')
    plt.close()

    # =============================================================================
    # Figure 2: Precision & Ranking Curves
    # =============================================================================
    print("Generating Figure 2: Precision & Ranking Curves...")
    fig2, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 0.9))
    axA, axB, axC, axD = axes.flatten()

    # --- Panel A: Precision Curve ---
    thresholds_prec = [1, 5, 10, 20, 50]
    for tool in tool_order:
        values = [benchmarker.metrics[tool]['accuracy'].get(f'precision_at_{t}', 0) for t in thresholds_prec]
        axA.plot(thresholds_prec, values, marker='o', markersize=3, linewidth=1, label=tool_labels[tool], color=tool_colors.get(tool, None))
    add_panel_label(axA, 'A')
    axA.set_xlabel('Rank Threshold (k)', fontsize=6)
    axA.set_ylabel('Mean Precision@k', fontsize=6)
    axA.set_xticks(thresholds_prec)
    axA.set_xscale('log')
    axA.legend(loc='upper right', fontsize=5)
    axA.grid(True, alpha=0.3, linewidth=0.5)

    # --- Panel B: F1 and Recall (Line Plot) ---
    f1_scores = [benchmarker.metrics[tool]['accuracy'].get('f1_score', 0) for tool in tool_order]
    recall_scores = [benchmarker.metrics[tool]['accuracy'].get('recall', 0) for tool in tool_order]
    x = np.arange(len(tool_order))
    axB.plot(x, f1_scores, marker='o', linestyle='-', label='F1 Score', color='#2196F3', markersize=4)
    axB.plot(x, recall_scores, marker='s', linestyle='--', label='Recall', color='#FF9800', markersize=4)
    add_panel_label(axB, 'B')
    axB.set_ylabel('Score', fontsize=6)
    axB.set_xticks(x)
    axB.set_xticklabels([tool_labels[t] for t in tool_order], fontsize=6, rotation=45, ha='right', fontweight='bold')
    axB.legend(fontsize=5, loc='best')
    axB.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    axB.set_ylim(0, 1.0)
    
    # --- Panel C: ROC/AUC Curve ---
    for tool in tool_order:
        auc_score = benchmarker.metrics[tool]['accuracy'].get('auc', np.nan)
        if not np.isnan(auc_score):
            # Recreate data for plotting
            all_scores, all_labels = [], []
            for sample_id, true_gene in ground_truth.items():
                if sample_id in benchmarker.tool_data[tool]:
                    ranked_genes = benchmarker.tool_data[tool][sample_id]
                    for i, gene in enumerate(ranked_genes):
                        all_scores.append(benchmarker.max_rank_length - i)
                        all_labels.append(1 if gene == true_gene else 0)
                    if true_gene not in ranked_genes:
                        all_scores.append(0)
                        all_labels.append(1)
                else:
                    all_scores.append(0)
                    all_labels.append(1)
            if len(all_labels) > 0 and len(set(all_labels)) > 1:
                fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
                axC.plot(fpr, tpr, color=tool_colors.get(tool, None), linewidth=1.5, label=f'{tool_labels[tool]} (AUC={auc_score:.3f})')
    axC.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    add_panel_label(axC, 'C')
    axC.set_xlabel('False Positive Rate', fontsize=6)
    axC.set_ylabel('True Positive Rate', fontsize=6)
    axC.legend(loc='lower right', fontsize=5)
    axC.grid(True, alpha=0.3, linewidth=0.5)

    # --- Panel D: ECDF Plot ---
    for tool in tool_order:
        ranks = benchmarker.metrics[tool]['distribution'].get('ecdf_ranks', [])
        if ranks:
            x_ranks = np.sort(ranks)
            y_ranks = np.arange(1, len(x_ranks) + 1) / len(x_ranks)
            axD.step(x_ranks, y_ranks, label=tool_labels[tool], linewidth=1.5, color=tool_colors.get(tool, None))
    add_panel_label(axD, 'D')
    axD.set_xlabel('Rank', fontsize=6)
    axD.set_ylabel('Cumulative Probability', fontsize=6)
    axD.legend(loc='lower right', fontsize=5)
    axD.grid(True, alpha=0.3, linewidth=0.5)
    axD.set_xlim(0, 50)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.35)
    plt.savefig(os.path.join(output_dir, 'Figure2_precision_curves.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'Figure2_precision_curves.png'), format='png')
    plt.close()

    # =============================================================================
    # Figure 3: Statistical Validation
    # =============================================================================
    print("Generating Figure 3: Statistical Validation...")
    fig3, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_INCHES, SINGLE_COLUMN_WIDTH * 1.1))
    axA, axB = axes.flatten()
    
    # --- Panel A: Bootstrap CIs ---
    metrics_list = ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']
    labels = ['Top-1', 'Top-5', 'Top-10']
    x_pos = np.arange(len(labels))
    for i, tool in enumerate(tool_order):
        if tool in benchmarker.statistical_tests['bootstrap_ci']:
            means, lowers, uppers = [], [], []
            for metric in metrics_list:
                ci_data = benchmarker.statistical_tests['bootstrap_ci'][tool][metric]
                means.append(ci_data.get('mean', np.nan))
                lowers.append(ci_data.get('lower', np.nan))
                uppers.append(ci_data.get('upper', np.nan))
            offset = (i - (len(tool_order) - 1) / 2) * 0.12
            yerr = [np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)]
            axA.errorbar(x_pos + offset, means, yerr=yerr, fmt='o', capsize=3, markersize=3, label=tool_labels[tool], color=tool_colors.get(tool, None), linewidth=1)
    add_panel_label(axA, 'A')
    axA.set_xlabel('Metric', fontsize=6)
    axA.set_ylabel('Accuracy (%) with 95% CI', fontsize=6)
    axA.set_xticks(x_pos)
    axA.set_xticklabels(labels, fontsize=5)
    axA.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, fontsize=5)
    axA.grid(True, alpha=0.3, linewidth=0.5)
    axA.set_ylim(-5, 100)

    # --- Panel B: Friedman Rank Distribution (Violin Plot) ---
    friedman_test = benchmarker.statistical_tests['friedman_test']
    if friedman_test and 'rank_matrix' in friedman_test and friedman_test['rank_matrix'] is not None:
        rank_matrix = friedman_test['rank_matrix']
        p_value = friedman_test.get('p_value', np.nan)
        df = pd.DataFrame(rank_matrix, columns=all_tools)
        df_melted = df.melt(var_name='Tool', value_name='Rank')
        violin_order = df.median().sort_values().index
        sns.violinplot(ax=axB, x='Tool', y='Rank', data=df_melted, order=violin_order, palette=tool_colors, inner='box', cut=0, linewidth=0.5)
        axB.set_xticklabels([tool_labels[t] for t in violin_order], rotation=45, ha="right", fontsize=6, fontweight='bold')
        axB.set_ylabel('Rank Distribution', fontsize=6)
        axB.set_xlabel('')
        axB.set_yscale('symlog', linthresh=10)
        axB.set_ylim(bottom=0)
        title_text = f"Friedman Test (p = {p_value:.3f})" if not np.isnan(p_value) else "Friedman Test"
        axB.set_title(title_text, fontsize=7)
        axB.grid(axis='y', linestyle='--', alpha=0.6)
    add_panel_label(axB, 'B')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_dir, 'Figure3_statistical_validation.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'Figure3_statistical_validation.png'), format='png')
    plt.close()
    
    print(f"Journal-compliant figures saved to {output_dir}/")

def main():
    """Main function to run the unified benchmark analysis and generate figures."""
    # --- USER: Define your file paths here ---
    base_dir = r"C:\Users\z5537966\OneDrive - UNSW\Desktop\new data\test\common_samples_only\Franklin\manual\final_combined_data\Testis\limited_samples\filtered_data"
    unified_acmg_path = os.path.join(base_dir, "ACMG_163new.csv")
    lirical_folder_path = base_dir
    manual_annotations_path = os.path.join(base_dir, "manual_163new.xlsx")
    
    output_dir = os.path.join(base_dir, "publication_figures_unified_analysis")
    
    print("\n===== BENCHMARKING ALL SAMPLES")
    
    # Initialize the benchmarker for a single, unified analysis
    benchmarker = VariantToolBenchmarker(
        acmg_path=unified_acmg_path,
        lirical_folder=lirical_folder_path,
        manual_path=manual_annotations_path,
        sample_subset=None,              # Use all available samples
        analysis_type='all_samples'
    )
    
    # Run the analysis pipeline
    benchmarker.load_data()
    benchmarker.calculate_metrics()
    benchmarker.perform_statistical_tests()
    
    # Generate the report and figures
    print("\n===== GENERATING REPORT AND JOURNAL-COMPLIANT FIGURES =====")
    
    report_path = os.path.join(output_dir, "unified_analysis_report.txt")
    benchmarker.generate_report(report_path)
    
    generate_figures(benchmarker, output_dir)
    
    print(f"\nAnalysis complete! Figures and report saved to: {output_dir}")

if __name__ == "__main__":
    main()
