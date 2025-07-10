variant_benchmarking.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Variant Classification Tools Benchmarking Script with Panel Generation

This script benchmarks variant classification tools against manual annotations,
generating publication-ready multi-panel figures combining Mendelian and cancer analyses.
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

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 14,              # Base font size
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,         # Axis labels
    'axes.titlesize': 18,         # Axis title
    'xtick.labelsize': 14,        # X-tick labels
    'ytick.labelsize': 14,        # Y-tick labels
    'legend.fontsize': 12,        # Legend text
    'figure.titlesize': 20        # Figure title
})
sns.set_context("paper")  # Use paper context for multi-panel figures

# Suppress warnings
warnings.filterwarnings('ignore')

class VariantToolBenchmarker:
    """Class to benchmark variant classification tools."""
    
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
            Type of analysis ('all', 'cancer', or 'mendelian')
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

    def load_data(self):
        """Load and preprocess all data sources."""
        print(f"Loading and preprocessing data for {self.analysis_type} analysis...")
        
        # Load unified ACMG data
        self.acmg_data = pd.read_csv(self.acmg_path)
        print(f"Loaded ACMG data: {self.acmg_data.shape[0]} rows")
        
        # Filter out excluded tools
        self.acmg_filtered = self.acmg_data[~self.acmg_data['tool'].isin(self.exclude_tools)]
        print(f"After filtering {', '.join(self.exclude_tools)}: {self.acmg_filtered.shape[0]} rows")
        
        # Load manual annotations (ground truth)
        self.manual_data = pd.read_excel(self.manual_path)
        print(f"Loaded manual annotations: {self.manual_data.shape[0]} rows")
        # Fix column names if needed
        if 'sample_id' in self.manual_data.columns and 'Sample id' not in self.manual_data.columns:
            self.manual_data = self.manual_data.rename(columns={'sample_id': 'Sample id'})
            print("Renamed 'sample_id' column to 'Sample id'")
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
        
        # Analyze Franklin data to understand its characteristics
        self._check_franklin_data()
        
        # Create a unified dataset by tool
        self._prepare_tool_data()
        print("Data loading and preprocessing complete.")

    def _check_franklin_data(self):
        """
        Special function to check Franklin data and report its characteristics.
        This helps understand how Franklin prioritizes variants and what columns might be useful.
        """
        # Get Franklin data
        franklin_df = self.acmg_filtered[self.acmg_filtered['tool'] == 'franklin'].copy()
        
        if franklin_df.empty:
            print("No Franklin data found")
            return
        
        print(f"\nAnalyzing Franklin data: {len(franklin_df)} variants across {franklin_df['sample_id'].nunique()} samples")
        
        # Check classifications
        class_counts = franklin_df['classification'].value_counts()
        total_variants = len(franklin_df)
        
        print("\nClassification distribution:")
        for cls, count in class_counts.items():
            percentage = (count / total_variants) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")
        
        # Check available columns that might be useful for ranking
        print("\nPotentially useful columns for ranking:")
        important_keywords = ['rank', 'score', 'priority', 'impact', 'effect', 'consequence', 
                             'cadd', 'freq', 'maf', 'vaf', 'pathogenic']
        
        useful_columns = []
        for col in franklin_df.columns:
            if any(keyword in col.lower() for keyword in important_keywords):
                useful_columns.append(col)
        
        for col in useful_columns:
            # Get some statistics for this column
            try:
                if franklin_df[col].dtype in ['int64', 'float64']:
                    print(f"  {col}: numeric, range={franklin_df[col].min()}-{franklin_df[col].max()}, unique values={franklin_df[col].nunique()}")
                else:
                    print(f"  {col}: {franklin_df[col].dtype}, unique values={franklin_df[col].nunique()}")
            except:
                print(f"  {col}: unknown type")
        
        # Check variant counts per sample
        variant_counts = franklin_df.groupby('sample_id').size()
        print(f"\nVariant counts per sample: min={variant_counts.min()}, max={variant_counts.max()}, mean={variant_counts.mean():.1f}")
        
        # Check if all samples have the same number of variants
        if variant_counts.nunique() == 1:
            print(f"All samples have exactly {variant_counts.iloc[0]} variants - suggests data was pre-filtered")
        
        # Return useful columns for use in ranking
        return useful_columns

    def _load_lirical_data(self):
        """
        Load LIRICAL data from a single combined TSV file instead of multiple files.
        """
        lirical_file = os.path.join(self.lirical_folder, "LIRICAL_185.tsv")
        
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
            if 'tool' in lirical_data.columns:
                lirical_data['tool'] = lirical_data['tool'].str.lower()
            else:
                lirical_data['tool'] = 'lirical'
                
            print(f"Loaded LIRICAL data with {len(lirical_data)} rows across {lirical_data['sample_id'].nunique()} samples")
            return lirical_data
            
        except Exception as e:
            print(f"Error loading LIRICAL data: {e}")
            return None

    def _prepare_tool_data(self):
        """
        Updated function to prepare data for each tool using native ranking systems.
        This treats all tools similar to LIRICAL, preserving their native prioritization.
        """
        # Get list of tools (excluding specified tools)
        tools = self.acmg_filtered['tool'].unique().tolist()
        
        # Add LIRICAL if available
        if self.lirical_data is not None:
            tools.append('LIRICAL')
        
        print(f"Preparing data for tools: {', '.join(tools)}")
        
        # Get ground truth genes for reference
        ground_truth = {}
        for _, row in self.manual_data.iterrows():
            sample_id = row['Sample id']
            gene = row['hgnc_gene']
            ground_truth[sample_id] = gene
        
        # For each tool, create a dataset with ranking information
        for tool in tools:
            if tool != 'LIRICAL':
                # Get data for this tool
                tool_df = self.acmg_filtered[self.acmg_filtered['tool'] == tool].copy()
                
                # Check if we have data for this tool
                if tool_df.empty:
                    print(f"Warning: No data found for tool {tool}")
                    continue
                
                # Debug info
                print(f"Processing {tool} with {len(tool_df)} variants across {tool_df['sample_id'].nunique()} samples")
                
                # Create a mapping for each sample's gene rankings
                sample_gene_rankings = {}
                
                # Process each sample separately
                for sample_id, group in tool_df.groupby('sample_id'):
                    # Skip if sample isn't in ground truth
                    if sample_id not in ground_truth:
                        continue
                    
                    try:
                        # For all tools, use their native ranking (similar to LIRICAL)
                        # Check for tool-specific ranking columns
                        if tool == 'franklin':
                            # Check if there's a ranking column specific to Franklin
                            franklin_rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority']) and 'franklin' in col.lower()]
                            
                            if franklin_rank_cols:
                                # Use Franklin's own ranking column
                                rank_col = franklin_rank_cols[0]
                                print(f"Using {rank_col} for Franklin ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # Preserve original order - Franklin likely already sorted variants by importance
                                # This respects Franklin's native prioritization
                                sorted_df = group
                        
                        elif tool == 'genebe':
                            # Check for GeneBE ranking columns
                            genebe_rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority']) and 'genebe' in col.lower()]
                            
                            if genebe_rank_cols:
                                rank_col = genebe_rank_cols[0]
                                print(f"Using {rank_col} for GeneBE ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # If no specific ranking column, use order as is
                                sorted_df = group
                        
                        elif tool == 'intervar':
                            # Check for InterVar ranking columns
                            intervar_rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority']) and 'intervar' in col.lower()]
                            
                            if intervar_rank_cols:
                                rank_col = intervar_rank_cols[0]
                                print(f"Using {rank_col} for InterVar ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # If no specific ranking column, use order as is
                                sorted_df = group
                        
                        elif tool == 'tapes':
                            # Check for TAPES ranking columns
                            tapes_rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority']) and 'tapes' in col.lower()]
                            
                            if tapes_rank_cols:
                                rank_col = tapes_rank_cols[0]
                                print(f"Using {rank_col} for TAPES ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # If no specific ranking column, use order as is
                                sorted_df = group
                        
                        elif tool == 'cpsr':
                            # Check for CPSR ranking columns
                            cpsr_rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority']) and 'cpsr' in col.lower()]
                            
                            if cpsr_rank_cols:
                                rank_col = cpsr_rank_cols[0]
                                print(f"Using {rank_col} for CPSR ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # If no specific ranking column, use order as is
                                sorted_df = group
                        
                        else:
                            # For any other tool, check for ranking columns
                            rank_cols = [col for col in group.columns if any(x in col.lower() for x in ['rank', 'score', 'priority'])]
                            
                            if rank_cols:
                                rank_col = rank_cols[0]
                                print(f"Using {rank_col} for {tool} ranking")
                                sorted_df = group.sort_values(by=[rank_col], ascending=True if 'rank' in rank_col.lower() else False)
                            else:
                                # If no ranking column found, preserve original order
                                sorted_df = group
                        
                        # Extract sorted genes
                        sorted_genes = []
                        seen = set()
                        for gene in sorted_df['hgnc_gene'].tolist():
                            if gene not in seen:
                                sorted_genes.append(gene)
                                seen.add(gene)
                        
                        # Store rankings
                        sample_gene_rankings[sample_id] = sorted_genes
                        
                        # Debug info for top genes
                        true_gene = ground_truth[sample_id]
                        if true_gene in sorted_genes:
                            rank = sorted_genes.index(true_gene) + 1
                            if rank <= 5:  # Only log for top-ranked matches
                                print(f"  {tool} - {sample_id}: Found {true_gene} at rank {rank}/{len(sorted_genes)}")
                    
                    except Exception as e:
                        print(f"Error processing {tool} data for sample {sample_id}: {e}")
                        # Fallback: just use genes as they appear
                        sorted_genes = group['hgnc_gene'].unique().tolist()
                        sample_gene_rankings[sample_id] = sorted_genes
                
                # Store all sample rankings for this tool
                self.tool_data[tool] = sample_gene_rankings
                
                # Calculate and display summary metrics
                self._calculate_tool_summary(tool, sample_gene_rankings, ground_truth)
            
            else:  # Special handling for LIRICAL
                # Process LIRICAL data if available
                if self.lirical_data is not None:
                    print(f"Processing LIRICAL data with {len(self.lirical_data)} rows across {self.lirical_data['sample_id'].nunique()} samples")
                    
                    # No need to check for rank column as it's already in the unified data
                    sample_gene_rankings = {}
                    
                    for sample_id, group in self.lirical_data.groupby('sample_id'):
                        try:
                            # Sort by rank (ascending = lowest rank first)
                            sorted_genes = group.sort_values(by=['rank'], ascending=True)['hgnc_gene'].tolist()
                            
                            # Remove duplicates
                            seen = set()
                            sorted_genes = [g for g in sorted_genes if not (g in seen or seen.add(g))]
                            
                            sample_gene_rankings[sample_id] = sorted_genes
                            
                            # Check if ground truth gene is found
                            if sample_id in ground_truth:
                                true_gene = ground_truth[sample_id]
                                if true_gene in sorted_genes:
                                    rank = sorted_genes.index(true_gene) + 1
                                    if rank <= 5:  # Only log for top-ranked matches
                                        print(f"  LIRICAL - {sample_id}: Found {true_gene} at rank {rank}/{len(sorted_genes)}")
                                        
                        except Exception as e:
                            print(f"Error processing LIRICAL data for sample {sample_id}: {e}")
                    
                    # Store LIRICAL rankings
                    self.tool_data[tool] = sample_gene_rankings
                    
                    # Calculate and display summary metrics
                    self._calculate_tool_summary(tool, sample_gene_rankings, ground_truth)

    def _calculate_tool_summary(self, tool, sample_gene_rankings, ground_truth):
        """Calculate and display summary metrics for a tool."""
        found_count = 0
        top1_count = 0
        top5_count = 0
        top10_count = 0
        for sample_id, true_gene in ground_truth.items():
            if sample_id in sample_gene_rankings:
                if true_gene in sample_gene_rankings[sample_id]:
                    found_count += 1
                    rank = sample_gene_rankings[sample_id].index(true_gene) + 1
                    if rank == 1:
                        top1_count += 1
                    if rank <= 5:
                        top5_count += 1
                    if rank <= 10:
                        top10_count += 1
        
        total_samples = len([s for s in ground_truth.keys() if s in sample_gene_rankings])
        print(f"Tool {tool} summary:")
        print(f"  Found {found_count} out of {total_samples} ground truth genes overall")
        if total_samples > 0:
            print(f"  Top-1 accuracy: {top1_count/total_samples*100:.1f}%")
            print(f"  Top-5 accuracy: {top5_count/total_samples*100:.1f}%")
            print(f"  Top-10 accuracy: {top10_count/total_samples*100:.1f}%")
            print(f"  Not found: {total_samples - found_count} ({(total_samples - found_count)/total_samples*100:.1f}%)")

    def calculate_metrics(self):
        """Calculate all performance metrics for each tool."""
        print("Calculating performance metrics...")
        
        for tool, rankings in self.tool_data.items():
            print(f"Processing tool: {tool}")
            
            tool_metrics = {
                'ranking': {},
                'filtering': {},
                'accuracy': {}
            }
            
            # Calculate ranking metrics
            tool_metrics['ranking'] = self._calculate_ranking_metrics(rankings)
            
            # Calculate filtering metrics
            tool_metrics['filtering'] = self._calculate_filtering_metrics(rankings)
            
            # Calculate accuracy metrics
            tool_metrics['accuracy'] = self._calculate_accuracy_metrics(rankings)
            
            # Store metrics for this tool
            self.metrics[tool] = tool_metrics
        
        print("Performance metrics calculation complete.")

    def _calculate_ranking_metrics(self, rankings):
        """
        Calculate ranking metrics:
        - Top-rank percentage
        - Top-N percentage (N=5,10,20,30,40,50)
        - Mean and median rank (only for found cases)
        - Min/max rank (only for found cases)
        """
        metrics = {}
        
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        found_ranks = []  # Only collect ranks where the gene is found
        top_1_count = 0
        top_n_counts = {5: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
        total_samples = 0
        
        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                total_samples += 1
                ranked_genes = rankings[sample_id]
                
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                    found_ranks.append(rank)  # Collect rank only if found
                    if rank == 1:
                        top_1_count += 1
                    for n in top_n_counts.keys():
                        if rank <= n:
                            top_n_counts[n] += 1
        
        if total_samples > 0:
            metrics['top_rank_percentage'] = (top_1_count / total_samples) * 100
            for n, count in top_n_counts.items():
                metrics[f'top_{n}_percentage'] = (count / total_samples) * 100
        else:
            metrics['top_rank_percentage'] = np.nan
            for n in top_n_counts.keys():
                metrics[f'top_{n}_percentage'] = np.nan
        
        if found_ranks:
            metrics['mean_rank'] = np.mean(found_ranks)
            metrics['median_rank'] = np.median(found_ranks)
            metrics['min_rank'] = np.min(found_ranks)
            metrics['max_rank'] = np.max(found_ranks)
        else:
            metrics['mean_rank'] = np.nan
            metrics['median_rank'] = np.nan
            metrics['min_rank'] = np.nan
            metrics['max_rank'] = np.nan
        
        return metrics

    def _calculate_filtering_metrics(self, rankings):
        """
        Calculate filtering metrics:
        - Filtered-out rate
        - Retention rate
        """
        metrics = {}
        
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        filtered_out_count = 0
        retained_count = 0
        total_samples = 0
        
        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                total_samples += 1
                ranked_genes = rankings[sample_id]
                if true_gene in ranked_genes:
                    retained_count += 1
                else:
                    filtered_out_count += 1
        
        if total_samples > 0:
            metrics['filtered_out_rate'] = (filtered_out_count / total_samples) * 100
            metrics['retention_rate'] = (retained_count / total_samples) * 100
        else:
            metrics['filtered_out_rate'] = np.nan
            metrics['retention_rate'] = np.nan
        
        return metrics

    def _calculate_accuracy_metrics(self, rankings):
        """
        Calculate accuracy metrics:
        - Precision at different rank thresholds
        - Recall (sensitivity)
        - Specificity
        - F1 score
        """
        metrics = {}
        
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        
        rank_thresholds = [1, 5, 10, 20, 50]
        
        # Calculate precision at different thresholds using global precision
        for threshold in rank_thresholds:
            true_positives = 0
            total_predictions = 0
            
            for sample_id, true_gene in ground_truth.items():
                if sample_id in rankings:
                    ranked_genes = rankings[sample_id]
                    # Take top k genes (or all if less than k)
                    top_k_genes = ranked_genes[:min(threshold, len(ranked_genes))]
                    
                    total_predictions += len(top_k_genes)
                    if true_gene in top_k_genes:
                        true_positives += 1
            
            if total_predictions > 0:
                precision = true_positives / total_predictions
            else:
                precision = 0.0
                
            metrics[f'precision_at_{threshold}'] = precision
        
        true_positives = 0
        false_negatives = 0
        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                if true_gene in ranked_genes:
                    true_positives += 1
                else:
                    false_negatives += 1
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = np.nan
        metrics['recall'] = recall
        
        true_negatives = 0
        false_positives = 0
        for sample_id, true_gene in ground_truth.items():
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                true_negatives += len(set(self.acmg_filtered['hgnc_gene'].unique()) - set(ranked_genes))
                false_positives += len(ranked_genes) - (1 if true_gene in ranked_genes else 0)
        if true_negatives + false_positives > 0:
            specificity = true_negatives / (true_negatives + false_positives)
        else:
            specificity = np.nan
        metrics['specificity'] = specificity
        
        if recall + metrics['precision_at_1'] > 0:
            f1 = 2 * (metrics['precision_at_1'] * recall) / (metrics['precision_at_1'] + recall)
        else:
            f1 = np.nan
        metrics['f1_score'] = f1
        
        return metrics

    def perform_statistical_tests(self):
        """Perform statistical tests without generating individual visualizations."""
        print(f"Performing statistical tests for {self.analysis_type} analysis...")
        
        self.statistical_tests = {
            'bootstrap_ci': {},
            'friedman_test': {},
            'nemenyi_test': {},
            'fisher_exact': {}
        }
        
        tools = list(self.tool_data.keys())
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        sample_ids = sorted(list(set(ground_truth.keys()) & set().union(*[set(self.tool_data[t].keys()) for t in tools])))
        n_samples = len(sample_ids)
        print(f"Samples for statistical tests: {n_samples}")
        
        if n_samples < 10:
            print("Warning: Small sample size (<10) may affect test reliability")
        
        # Bootstrap Confidence Intervals
        print("Calculating bootstrap CIs for Top-1, Top-5, Top-10...")
        n_resamples = 1000
        metrics_list = ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']
        
        for tool in tools:
            self.statistical_tests['bootstrap_ci'][tool] = {}
            for metric in metrics_list:
                samples = []
                threshold = 1 if metric == 'top_rank_percentage' else int(metric.split('_')[1])
                for sample_id, true_gene in ground_truth.items():
                    if sample_id in self.tool_data[tool]:
                        ranked_genes = self.tool_data[tool][sample_id]
                        found = true_gene in ranked_genes and ranked_genes.index(true_gene) < threshold
                        samples.append(1 if found else 0)
                
                if len(samples) < 5:
                    print(f"Warning: Too few samples ({len(samples)}) for {tool} {metric} CI")
                    self.statistical_tests['bootstrap_ci'][tool][metric] = {'lower': np.nan, 'upper': np.nan, 'mean': np.nan}
                    continue
                
                bootstrap_results = [np.mean(resample(samples)) * 100 for _ in range(n_resamples)]
                lower, upper = np.percentile(bootstrap_results, [2.5, 97.5])
                self.statistical_tests['bootstrap_ci'][tool][metric] = {
                    'lower': lower,
                    'upper': upper,
                    'mean': np.mean(bootstrap_results)
                }
        
        # Friedman & Nemenyi Tests
        print("Performing Friedman and Nemenyi tests...")
        if n_samples >= 5 and len(tools) >= 2:
            rank_matrix = []
            for sample_id in sample_ids:
                sample_ranks = []
                for tool in tools:
                    if sample_id in self.tool_data[tool]:
                        ranked_genes = self.tool_data[tool][sample_id]
                        rank = ranked_genes.index(ground_truth[sample_id]) + 1 if ground_truth[sample_id] in ranked_genes else len(ranked_genes) + 1
                    else:
                        avg_rank = np.mean([len(self.tool_data[tool].get(sid, [])) + 1 
                                          for sid in self.tool_data[tool].keys() 
                                          if ground_truth.get(sid) not in self.tool_data[tool].get(sid, [])])
                        rank = avg_rank if not np.isnan(avg_rank) else 1000
                    sample_ranks.append(rank)
                rank_matrix.append(sample_ranks)
            
            rank_matrix = np.array(rank_matrix)
            try:
                statistic, p_value = friedmanchisquare(*rank_matrix.T)
                self.statistical_tests['friedman_test'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                print(f"Friedman test: statistic={statistic:.2f}, p-value={p_value:.4f}")
                
                if p_value < 0.05:
                    nemenyi_result = sp.posthoc_nemenyi_friedman(rank_matrix)
                    self.statistical_tests['nemenyi_test'] = {'matrix': nemenyi_result, 'tool_pairs': {}}
                    for i, tool1 in enumerate(tools):
                        for j, tool2 in enumerate(tools):
                            if i < j:
                                p_val = nemenyi_result.iloc[i, j]
                                self.statistical_tests['nemenyi_test']['tool_pairs'][(tool1, tool2)] = {
                                    'p_value': p_val,
                                    'significant': p_val < 0.05
                                }
                                if p_val < 0.05:
                                    print(f"Significant difference: {tool1} vs {tool2}, p={p_val:.4f}")
            except Exception as e:
                print(f"Friedman test error: {e}")
        else:
            print("Need >=5 samples and >=2 tools for Friedman test")
        
        # Fisher's Exact Test
        print("Performing Fisher's exact test with Bonferroni correction...")
        self.statistical_tests['fisher_exact'] = {'top_1': {}, 'top_5': {}, 'top_10': {}}
        thresholds = [1, 5, 10]
        keys = ['top_1', 'top_5', 'top_10']
        all_p_values = []
        all_pairs = []
        
        for threshold, key in zip(thresholds, keys):
            for i, tool1 in enumerate(tools):
                for j, tool2 in enumerate(tools):
                    if i < j:
                        table = np.zeros((2, 2), dtype=int)
                        count = 0
                        for sample_id, true_gene in ground_truth.items():
                            if sample_id in self.tool_data[tool1] and sample_id in self.tool_data[tool2]:
                                found1 = true_gene in self.tool_data[tool1][sample_id] and self.tool_data[tool1][sample_id].index(true_gene) < threshold
                                found2 = true_gene in self.tool_data[tool2][sample_id] and self.tool_data[tool2][sample_id].index(true_gene) < threshold
                                table[int(found1), int(found2)] += 1
                                count += 1
                        
                        if count < 10:
                            print(f"Warning: Low sample count ({count}) for {tool1} vs {tool2} at top-{threshold}")
                        if np.sum(table) == 0:
                            continue
                        
                        try:
                            odds_ratio, p_value = fisher_exact(table)
                            all_p_values.append(p_value)
                            all_pairs.append((tool1, tool2, threshold, key, table, odds_ratio))
                        except Exception as e:
                            print(f"Fisher's test error for {tool1} vs {tool2}: {e}")
        
        if all_p_values:
            corrected_p = multipletests(all_p_values, method='bonferroni')[1]
            for (tool1, tool2, threshold, key, table, odds_ratio), p in zip(all_pairs, corrected_p):
                tool1_acc = self.metrics[tool1]['ranking'].get('top_rank_percentage' if threshold == 1 else f'top_{threshold}_percentage', 0)
                tool2_acc = self.metrics[tool2]['ranking'].get('top_rank_percentage' if threshold == 1 else f'top_{threshold}_percentage', 0)
                better = tool1 if tool1_acc > tool2_acc else tool2
                self.statistical_tests['fisher_exact'][key][(tool1, tool2)] = {
                    'odds_ratio': odds_ratio,
                    'p_value': p,
                    'significant': p < 0.05,
                    'contingency_table': table.tolist(),
                    'better_tool': better
                }
                if p < 0.05:
                    print(f"Significant: top-{threshold}, {tool1} vs {tool2}, corrected p={p:.4f}, {better} better")

    def calculate_auc(self):
        """Calculate corrected AUC for each tool that penalizes filtered-out genes."""
        print(f"Calculating corrected AUC for each tool ({self.analysis_type} analysis)...")
        
        ground_truth = {row['Sample id']: row['hgnc_gene'] for _, row in self.manual_data.iterrows()}
        self.auc_results = {}
        
        for tool, rankings in self.tool_data.items():
            print(f"Calculating AUC for {tool}...")
            
            all_scores = []
            all_labels = []
            
            # First pass: determine maximum list length across all samples for this tool
            max_length = 0
            for sample_id in ground_truth.keys():
                if sample_id in rankings:
                    max_length = max(max_length, len(rankings[sample_id]))
            
            # If no data for this tool, skip
            if max_length == 0:
                self.auc_results[tool] = np.nan
                continue
            
            for sample_id, true_gene in ground_truth.items():
                if sample_id in rankings:
                    ranked_genes = rankings[sample_id]
                    
                    # Process all genes in the ranking
                    for i, gene in enumerate(ranked_genes):
                        score = max_length - i  # Higher rank = higher score
                        label = 1 if gene == true_gene else 0
                        all_scores.append(score)
                        all_labels.append(label)
                    
                    # CRITICAL: If causative gene is not in the ranking (filtered out),
                    # assign it the worst possible score (0)
                    if true_gene not in ranked_genes:
                        all_scores.append(0)  # Worst score for filtered-out gene
                        all_labels.append(1)  # This IS the causative gene
            
            # Calculate AUC if we have both positive and negative cases
            if len(set(all_labels)) == 2 and len(all_scores) > 0:
                try:
                    auc_score = metrics.roc_auc_score(all_labels, all_scores)
                    self.auc_results[tool] = auc_score
                    print(f"  {tool}: AUC = {auc_score:.4f}")
                except Exception as e:
                    print(f"  Error calculating AUC for {tool}: {e}")
                    self.auc_results[tool] = np.nan
            else:
                print(f"  Cannot calculate AUC for {tool}: insufficient data")
                self.auc_results[tool] = np.nan
        
        print("Corrected AUC calculation complete.")

    def generate_report(self, output_file='benchmark_report.txt'):
        """
        Generate a comprehensive report of the results.
        
        Parameters:
        -----------
        output_file : str
            Path to save the report
        """
        print("Generating report...")
        
        with open(output_file, 'w') as f:
            f.write("========================================================\n")
            f.write(f"   VARIANT CLASSIFICATION TOOLS BENCHMARKING REPORT ({self.analysis_type.upper()})    \n")
            f.write("========================================================\n\n")
            
            # Summary of tools analyzed
            f.write("TOOLS ANALYZED\n")
            f.write("-------------\n")
            for tool in self.tool_data.keys():
                f.write(f"- {tool}\n")
            f.write("\n")
            
            # Ranking Metrics
            f.write("RANKING METRICS\n")
            f.write("--------------\n")
            metrics = ['Top-Rank %', 'Top-5 %', 'Top-10 %', 'Top-20 %', 'Mean Rank', 'Median Rank']
            f.write(f"{'Tool':<15}")
            for metric in metrics:
                f.write(f"{metric:<15}")
            f.write("\n")
            f.write("-" * (15 * (len(metrics) + 1)) + "\n")
            for tool in self.tool_data.keys():
                f.write(f"{tool:<15}")
                value = self.metrics[tool]['ranking'].get('top_rank_percentage', np.nan)
                f.write(f"{value:,.2f}%{' ' * 8}")
                value = self.metrics[tool]['ranking'].get('top_5_percentage', np.nan)
                f.write(f"{value:,.2f}%{' ' * 8}")
                value = self.metrics[tool]['ranking'].get('top_10_percentage', np.nan)
                f.write(f"{value:,.2f}%{' ' * 8}")
                value = self.metrics[tool]['ranking'].get('top_20_percentage', np.nan)
                f.write(f"{value:,.2f}%{' ' * 8}")
                value = self.metrics[tool]['ranking'].get('mean_rank', np.nan)
                f.write(f"{value:,.2f}{' ' * 10}")
                value = self.metrics[tool]['ranking'].get('median_rank', np.nan)
                f.write(f"{value:,.2f}{' ' * 10}")
                f.write("\n")
            f.write("\n")
            
            # Filtering Performance
            f.write("FILTERING PERFORMANCE\n")
            f.write("--------------------\n")
            metrics = ['Filtered-Out Rate', 'Retention Rate']
            f.write(f"{'Tool':<15}")
            for metric in metrics:
                f.write(f"{metric:<20}")
            f.write("\n")
            f.write("-" * (15 + 20 * len(metrics)) + "\n")
            for tool in self.tool_data.keys():
                f.write(f"{tool:<15}")
                value = self.metrics[tool]['filtering'].get('filtered_out_rate', np.nan)
                f.write(f"{value:,.2f}%{' ' * 13}")
                value = self.metrics[tool]['filtering'].get('retention_rate', np.nan)
                f.write(f"{value:,.2f}%{' ' * 13}")
                f.write("\n")
            f.write("\n")
            
            # Accuracy Metrics
            f.write("ACCURACY METRICS\n")
            f.write("---------------\n")
            metrics = ['Precision@1', 'Precision@10', 'Recall', 'F1 Score']
            f.write(f"{'Tool':<15}")
            for metric in metrics:
                f.write(f"{metric:<15}")
            f.write("\n")
            f.write("-" * (15 * (len(metrics) + 1)) + "\n")
            for tool in self.tool_data.keys():
                f.write(f"{tool:<15}")
                value = self.metrics[tool]['accuracy'].get('precision_at_1', np.nan)
                f.write(f"{value:,.4f}{' ' * 8}")
                value = self.metrics[tool]['accuracy'].get('precision_at_10', np.nan)
                f.write(f"{value:,.4f}{' ' * 8}")
                value = self.metrics[tool]['accuracy'].get('recall', np.nan)
                f.write(f"{value:,.4f}{' ' * 8}")
                value = self.metrics[tool]['accuracy'].get('f1_score', np.nan)
                f.write(f"{value:,.4f}{' ' * 8}")
                f.write("\n")
            f.write("\n")
            
            # AUC Results
            if hasattr(self, 'auc_results'):
                f.write("AUC SCORES\n")
                f.write("----------\n")
                f.write(f"{'Tool':<15}{'AUC Score':<15}\n")
                f.write("-" * 30 + "\n")
                sorted_tools = sorted(self.auc_results.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)
                for tool, auc_score in sorted_tools:
                    if np.isnan(auc_score):
                        f.write(f"{tool:<15}{'N/A':<15}\n")
                    else:
                        f.write(f"{tool:<15}{auc_score:<15.4f}\n")
                f.write("\n")
            
            # Add statistical tests
            self.update_report_with_statistics(f)
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("----------\n")
            f.write("Based on the benchmarking metrics, the tools can be ranked as follows:\n\n")
            tool_ranking = {tool: self.metrics[tool]['ranking'].get('top_10_percentage', 0) for tool in self.tool_data.keys()}
            sorted_tools = sorted(tool_ranking.items(), key=lambda x: x[1], reverse=True)
            for i, (tool, value) in enumerate(sorted_tools):
                f.write(f"{i+1}. {tool} (Top-10: {value:.2f}%)\n")
            f.write("\nRecommendations:\n")
            if 'cpsr' in self.tool_data and 'LIRICAL' in self.tool_data:
                f.write("1. For cancer samples, CPSR should be the first-line tool, with LIRICAL as complement\n")
                f.write("2. For non-cancer samples, LIRICAL should be the primary tool\n")
            elif 'LIRICAL' in self.tool_data:
                f.write("1. LIRICAL shows the strongest overall performance across metrics\n")
            f.write("3. Consider using multiple tools in combination to maximize variant identification\n")
            f.write("4. For specific applications, consider the strengths of each tool highlighted in this report\n")
            f.write("5. Franklin performs well at finding pathogenic variants but may need alternative ranking metrics\n")
        
        print(f"Report generated and saved to {output_file}")

    def update_report_with_statistics(self, f):
        """
        Update the report with statistical test results, including bootstrap CIs for top-1, top-5, top-10.
        
        Parameters:
        -----------
        f : file object
            The open file to write the report to
        """
        f.write("STATISTICAL TESTS\n")
        f.write("----------------\n")
    
        # Friedman Test
        if 'friedman_test' in self.statistical_tests:
            f.write("Friedman Test Results (Ranking Performance):\n")
            friedman = self.statistical_tests['friedman_test']
            f.write(f"  Statistic: {friedman.get('statistic', 'N/A'):.4f}\n")
            f.write(f"  p-value: {friedman.get('p_value', 'N/A'):.4f}\n")
            f.write(f"  Significant: {'Yes' if friedman.get('significant', False) else 'No'}\n\n")
    
        # Nemenyi Post-hoc Test
        if 'nemenyi_test' in self.statistical_tests and 'tool_pairs' in self.statistical_tests['nemenyi_test']:
            f.write("Nemenyi Post-hoc Test Results (Ranking Performance):\n")
            significant_pairs = [(t1, t2, data['p_value']) for (t1, t2), data in 
                                self.statistical_tests['nemenyi_test']['tool_pairs'].items() 
                                if data['significant']]
            if significant_pairs:
                f.write("  Significant tool pairs:\n")
                for t1, t2, p_val in sorted(significant_pairs, key=lambda x: x[2]):
                    tool1_acc = self.metrics[t1]['ranking'].get('top_rank_percentage', 0)
                    tool2_acc = self.metrics[t2]['ranking'].get('top_rank_percentage', 0)
                    better_tool = t1 if tool1_acc > tool2_acc else t2
                    f.write(f"    {t1} vs {t2}: p-value={p_val:.4f}, {better_tool} performs better\n")
            else:
                f.write("  No significant differences found between tool pairs.\n")
            f.write("\n")
    
        # Fisher's Exact Test
        if 'fisher_exact' in self.statistical_tests:
            f.write("Fisher's Exact Test Results:\n")
            for threshold, label in [('top_1', 'Top-1'), ('top_5', 'Top-5'), ('top_10', 'Top-10')]:
                if threshold in self.statistical_tests['fisher_exact']:
                    f.write(f"  {label} Comparison:\n")
                    significant_pairs = [(t1, t2, data['p_value'], data['odds_ratio']) 
                                        for (t1, t2), data in self.statistical_tests['fisher_exact'][threshold].items() 
                                        if data['significant']]
                    if significant_pairs:
                        f.write("    Significant tool pairs:\n")
                        for t1, t2, p_val, odds in sorted(significant_pairs, key=lambda x: x[2]):
                            better_tool = t1 if odds > 1 else t2
                            f.write(f"      {t1} vs {t2}: p-value={p_val:.4f}, {better_tool} performs better\n")
                    else:
                        f.write("    No significant differences found.\n")
            f.write("\n")
    
        # Bootstrap Confidence Intervals
        if 'bootstrap_ci' in self.statistical_tests:
            f.write("Bootstrap 95% Confidence Intervals:\n")
            f.write(f"{'Tool':<15}{'Top-1 (%)':<25}{'Top-5 (%)':<25}{'Top-10 (%)':<25}\n")
            f.write("-" * 100 + "\n")
            for tool in sorted(self.tool_data.keys()):
                if tool in self.statistical_tests['bootstrap_ci']:
                    ci_data = self.statistical_tests['bootstrap_ci'][tool]
                    f.write(f"{tool:<15}")
                    for metric in ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']:
                        if metric in ci_data:
                            data = ci_data[metric]
                            f.write(f"{data['mean']:.2f} ({data['lower']:.2f}-{data['upper']:.2f}){' '*(25-13)}")
                        else:
                            f.write(f"N/A{' '*(25-13)}")
                    f.write("\n")
            f.write("\n")


def generate_combined_panels(benchmarker_all, benchmarker_cancer, output_dir='combined_panels'):
    """Generate combined panel figures from two benchmarker instances."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define tool colors for consistency
    tool_colors = {
        'franklin': '#1f77b4',
        'Franklin': '#1f77b4',
        'genebe': '#ff7f0e',
        'Genebe': '#ff7f0e',
        'intervar': '#2ca02c',
        'Intervar': '#2ca02c',
        'InterVar': '#2ca02c',
        'tapes': '#d62728',
        'TAPES': '#d62728',
        'lirical': '#9467bd',
        'LIRICAL': '#9467bd',
        'cpsr': '#8c564b',
        'CPSR': '#8c564b'
    }
    
    # Get ground truth for both analyses
    ground_truth_all = {row['Sample id']: row['hgnc_gene'] for _, row in benchmarker_all.manual_data.iterrows()}
    ground_truth_cancer = {row['Sample id']: row['hgnc_gene'] for _, row in benchmarker_cancer.manual_data.iterrows()}
    
    # Figure 1: Radar plots and ranking accuracy
    print("Generating Figure 1: Performance overview...")
    fig1 = plt.figure(figsize=(20, 10))
    
    # Panel A: Radar plot for all samples
    ax1 = fig1.add_subplot(2, 2, 1, polar=True)
    tools_all = list(benchmarker_all.tool_data.keys())
    metrics_radar = ['Top-1 Accuracy (%)', 'Top-5 Accuracy (%)', 'Top-10 Accuracy (%)', 'Accuracy (%)']
    N = len(metrics_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_radar)
    ax1.set_ylim(0, 100)
    
    for tool in tools_all:
        values = [
            benchmarker_all.metrics[tool]['ranking'].get('top_rank_percentage', 0),
            benchmarker_all.metrics[tool]['ranking'].get('top_5_percentage', 0),
            benchmarker_all.metrics[tool]['ranking'].get('top_10_percentage', 0),
            benchmarker_all.metrics[tool]['accuracy'].get('recall', 0) * 100
        ]
        values += values[:1]
        ax1.plot(angles, values, linewidth=2, linestyle='solid', label=tool, color=tool_colors.get(tool, None))
        ax1.fill(angles, values, alpha=0.1, color=tool_colors.get(tool, None))
    
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True, shadow=True)
    
    # Panel B: Radar plot for cancer samples
    ax2 = fig1.add_subplot(2, 2, 2, polar=True)
    tools_cancer = list(benchmarker_cancer.tool_data.keys())
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_radar)
    ax2.set_ylim(0, 100)
    
    for tool in tools_cancer:
        values = [
            benchmarker_cancer.metrics[tool]['ranking'].get('top_rank_percentage', 0),
            benchmarker_cancer.metrics[tool]['ranking'].get('top_5_percentage', 0),
            benchmarker_cancer.metrics[tool]['ranking'].get('top_10_percentage', 0),
            benchmarker_cancer.metrics[tool]['accuracy'].get('recall', 0) * 100
        ]
        values += values[:1]
        ax2.plot(angles, values, linewidth=2, linestyle='solid', label=tool, color=tool_colors.get(tool, None))
        ax2.fill(angles, values, alpha=0.1, color=tool_colors.get(tool, None))
    
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True, shadow=True)
    
    # Panel C: Ranking accuracy for all samples
    ax3 = fig1.add_subplot(2, 2, 3)
    thresholds = [1, 5, 10, 20, 50]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    accuracy_data = np.zeros((len(tools_all), len(thresholds)))
    for i, tool in enumerate(tools_all):
        for j, threshold in enumerate(thresholds):
            metric_name = f'top_rank_percentage' if threshold == 1 else f'top_{threshold}_percentage'
            accuracy_data[i, j] = benchmarker_all.metrics[tool]['ranking'].get(metric_name, 0)
    
    width = 0.15
    x = np.arange(len(tools_all))
    for i in range(len(thresholds)):
        ax3.bar(x + (i - 2) * width, accuracy_data[:, i], width, label=f'Top-{thresholds[i]} Accuracy (%)', color=colors[i])
    
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=20)
    ax3.set_xlabel('Tool')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tools_all)
    ax3.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Ranking accuracy for cancer samples
    ax4 = fig1.add_subplot(2, 2, 4)
    accuracy_data_cancer = np.zeros((len(tools_cancer), len(thresholds)))
    for i, tool in enumerate(tools_cancer):
        for j, threshold in enumerate(thresholds):
            metric_name = f'top_rank_percentage' if threshold == 1 else f'top_{threshold}_percentage'
            accuracy_data_cancer[i, j] = benchmarker_cancer.metrics[tool]['ranking'].get(metric_name, 0)
    
    x_cancer = np.arange(len(tools_cancer))
    for i in range(len(thresholds)):
        ax4.bar(x_cancer + (i - 2) * width, accuracy_data_cancer[:, i], width, label=f'Top-{thresholds[i]} Accuracy (%)', color=colors[i])
    
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=20)
    ax4.set_xlabel('Tool')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(x_cancer)
    ax4.set_xticklabels(tools_cancer)
    ax4.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.90, hspace=0.3, wspace=0.4)  # Add space for legends
    plt.savefig(os.path.join(output_dir, 'Figure1_performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Retention rates and F1/Recall
    print("Generating Figure 2: Retention and balanced performance...")
    fig2 = plt.figure(figsize=(20, 10))
    
    # Panel A: Retention rate for all samples
    ax1 = fig2.add_subplot(2, 2, 1)
    retention_all = [benchmarker_all.metrics[tool]['filtering'].get('retention_rate', 0) for tool in tools_all]
    ax1.bar(tools_all, retention_all, color='#4CAF50')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=20)
    ax1.set_xlabel('Tool')
    ax1.set_ylabel('Retention Rate (%)')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Panel B: Retention rate for cancer samples
    ax2 = fig2.add_subplot(2, 2, 2)
    retention_cancer = [benchmarker_cancer.metrics[tool]['filtering'].get('retention_rate', 0) for tool in tools_cancer]
    ax2.bar(tools_cancer, retention_cancer, color='#4CAF50')
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=20)
    ax2.set_xlabel('Tool')
    ax2.set_ylabel('Retention Rate (%)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Panel C: F1 and Recall for all samples
    ax3 = fig2.add_subplot(2, 2, 3)
    f1_all = [benchmarker_all.metrics[tool]['accuracy'].get('f1_score', 0) for tool in tools_all]
    recall_all = [benchmarker_all.metrics[tool]['accuracy'].get('recall', 0) for tool in tools_all]
    x = np.arange(len(tools_all))
    width = 0.35
    ax3.bar(x - width/2, f1_all, width, label='F1 Score')
    ax3.bar(x + width/2, recall_all, width, label='Recall')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=20)
    ax3.set_xlabel('Tool')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tools_all)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Panel D: F1 and Recall for cancer samples
    ax4 = fig2.add_subplot(2, 2, 4)
    f1_cancer = [benchmarker_cancer.metrics[tool]['accuracy'].get('f1_score', 0) for tool in tools_cancer]
    recall_cancer = [benchmarker_cancer.metrics[tool]['accuracy'].get('recall', 0) for tool in tools_cancer]
    x_cancer = np.arange(len(tools_cancer))
    ax4.bar(x_cancer - width/2, f1_cancer, width, label='F1 Score')
    ax4.bar(x_cancer + width/2, recall_cancer, width, label='Recall')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=20)
    ax4.set_xlabel('Tool')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x_cancer)
    ax4.set_xticklabels(tools_cancer)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between subplots
    plt.savefig(os.path.join(output_dir, 'Figure2_retention_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Rank distributions, CDFs, and ROC curves
    print("Generating Figure 3: Ranking distributions and AUC...")
    fig3 = plt.figure(figsize=(20, 15))
    
    # Panel A: Rank distribution for all samples
    ax1 = fig3.add_subplot(3, 2, 1)
    rank_distributions = []
    categories = ['1st', '2nd-5th', '6th-10th', '>10th', 'FONP']
    colors = ['#4daf4a', '#ff7f00', '#377eb8', '#984ea3', '#e41a1c']
    
    for tool in tools_all:
        counts = {'1st': 0, '2nd-5th': 0, '6th-10th': 0, '>10th': 0, 'FONP': 0}
        for sample_id, true_gene in ground_truth_all.items():
            if sample_id in benchmarker_all.tool_data[tool]:
                ranked_genes = benchmarker_all.tool_data[tool][sample_id]
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                    if rank == 1:
                        counts['1st'] += 1
                    elif 2 <= rank <= 5:
                        counts['2nd-5th'] += 1
                    elif 6 <= rank <= 10:
                        counts['6th-10th'] += 1
                    else:
                        counts['>10th'] += 1
                else:
                    counts['FONP'] += 1
            else:
                counts['FONP'] += 1
        total = sum(counts.values())
        percentages = {k: (v / total * 100) for k, v in counts.items()}
        rank_distributions.append([percentages[cat] for cat in categories])
    
    data = np.array(rank_distributions).T
    bottoms = np.zeros(len(tools_all))
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax1.bar(tools_all, data[i], bottom=bottoms, label=cat, color=color)
        for j, tool in enumerate(tools_all):
            if data[i][j] > 5:
                ax1.text(j, bottoms[j] + data[i][j]/2, f"{data[i][j]:.1f}%", ha='center', va='center', fontsize=10)
        bottoms += data[i]
    
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=20)
    ax1.set_xlabel('Tool')
    ax1.set_ylabel('Percentage (%)')
    ax1.legend(title='Rank', loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Panel B: Rank distribution for cancer samples
    ax2 = fig3.add_subplot(3, 2, 2)
    rank_distributions_cancer = []
    
    for tool in tools_cancer:
        counts = {'1st': 0, '2nd-5th': 0, '6th-10th': 0, '>10th': 0, 'FONP': 0}
        for sample_id, true_gene in ground_truth_cancer.items():
            if sample_id in benchmarker_cancer.tool_data[tool]:
                ranked_genes = benchmarker_cancer.tool_data[tool][sample_id]
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                    if rank == 1:
                        counts['1st'] += 1
                    elif 2 <= rank <= 5:
                        counts['2nd-5th'] += 1
                    elif 6 <= rank <= 10:
                        counts['6th-10th'] += 1
                    else:
                        counts['>10th'] += 1
                else:
                    counts['FONP'] += 1
            else:
                counts['FONP'] += 1
        total = sum(counts.values())
        percentages = {k: (v / total * 100) for k, v in counts.items()}
        rank_distributions_cancer.append([percentages[cat] for cat in categories])
    
    data_cancer = np.array(rank_distributions_cancer).T
    bottoms_cancer = np.zeros(len(tools_cancer))
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax2.bar(tools_cancer, data_cancer[i], bottom=bottoms_cancer, label=cat, color=color)
        for j, tool in enumerate(tools_cancer):
            if data_cancer[i][j] > 5:
                ax2.text(j, bottoms_cancer[j] + data_cancer[i][j]/2, f"{data_cancer[i][j]:.1f}%", ha='center', va='center', fontsize=10)
        bottoms_cancer += data_cancer[i]
    
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=20)
    ax2.set_xlabel('Tool')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(title='Rank', loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Panel C: CDF for all samples
    ax3 = fig3.add_subplot(3, 2, 3)
    for tool in tools_all:
        ranks = []
        for sample_id, true_gene in ground_truth_all.items():
            if sample_id in benchmarker_all.tool_data[tool]:
                ranked_genes = benchmarker_all.tool_data[tool][sample_id]
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                else:
                    rank = len(ranked_genes) + 1
                ranks.append(rank)
        if ranks:
            x = np.sort(ranks)
            y = np.arange(1, len(x) + 1) / len(x)
            ax3.step(x, y, label=tool, linewidth=2, color=tool_colors.get(tool, None))
    
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=20)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)  # Limit x-axis for better visibility
    
    # Panel D: CDF for cancer samples
    ax4 = fig3.add_subplot(3, 2, 4)
    for tool in tools_cancer:
        ranks = []
        for sample_id, true_gene in ground_truth_cancer.items():
            if sample_id in benchmarker_cancer.tool_data[tool]:
                ranked_genes = benchmarker_cancer.tool_data[tool][sample_id]
                if true_gene in ranked_genes:
                    rank = ranked_genes.index(true_gene) + 1
                else:
                    rank = len(ranked_genes) + 1
                ranks.append(rank)
        if ranks:
            x = np.sort(ranks)
            y = np.arange(1, len(x) + 1) / len(x)
            ax4.step(x, y, label=tool, linewidth=2, color=tool_colors.get(tool, None))
    
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=20)
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Cumulative Probability')
    ax4.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)  # Limit x-axis for better visibility
    
    # Panel E: ROC curves for all samples
    ax5 = fig3.add_subplot(3, 2, 5)
    for tool, rankings in benchmarker_all.tool_data.items():
        all_scores = []
        all_labels = []
        
        max_length = 0
        for sample_id in ground_truth_all.keys():
            if sample_id in rankings:
                max_length = max(max_length, len(rankings[sample_id]))
        
        if max_length == 0:
            continue
        
        for sample_id, true_gene in ground_truth_all.items():
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                for i, gene in enumerate(ranked_genes):
                    score = max_length - i
                    label = 1 if gene == true_gene else 0
                    all_scores.append(score)
                    all_labels.append(label)
                if true_gene not in ranked_genes:
                    all_scores.append(0)
                    all_labels.append(1)
        
        if len(set(all_labels)) == 2 and len(all_scores) > 0:
            try:
                fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
                auc_score = metrics.auc(fpr, tpr)
                ax5.plot(fpr, tpr, color=tool_colors.get(tool, None), linewidth=2.5, 
                        label=f'{tool} (AUC = {auc_score:.3f})')
            except:
                pass
    
    ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax5.set_title('E', loc='left', fontweight='bold', fontsize=20)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3)
    
    # Panel F: ROC curves for cancer samples
    ax6 = fig3.add_subplot(3, 2, 6)
    for tool, rankings in benchmarker_cancer.tool_data.items():
        all_scores = []
        all_labels = []
        
        max_length = 0
        for sample_id in ground_truth_cancer.keys():
            if sample_id in rankings:
                max_length = max(max_length, len(rankings[sample_id]))
        
        if max_length == 0:
            continue
        
        for sample_id, true_gene in ground_truth_cancer.items():
            if sample_id in rankings:
                ranked_genes = rankings[sample_id]
                for i, gene in enumerate(ranked_genes):
                    score = max_length - i
                    label = 1 if gene == true_gene else 0
                    all_scores.append(score)
                    all_labels.append(label)
                if true_gene not in ranked_genes:
                    all_scores.append(0)
                    all_labels.append(1)
        
        if len(set(all_labels)) == 2 and len(all_scores) > 0:
            try:
                fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
                auc_score = metrics.auc(fpr, tpr)
                ax6.plot(fpr, tpr, color=tool_colors.get(tool, None), linewidth=2.5, 
                        label=f'{tool} (AUC = {auc_score:.3f})')
            except:
                pass
    
    ax6.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax6.set_title('F', loc='left', fontweight='bold', fontsize=20)
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.35, wspace=0.3)  # Add space for legends
    plt.savefig(os.path.join(output_dir, 'Figure3_distributions_auc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Precision curves and rank comparisons
    print("Generating Figure 4: Precision and ranking efficiency...")
    fig4 = plt.figure(figsize=(20, 10))
    
    # Panel A: Precision curves for all samples
    ax1 = fig4.add_subplot(2, 2, 1)
    thresholds = [1, 5, 10, 20, 50]
    for tool in tools_all:
        values = [benchmarker_all.metrics[tool]['accuracy'].get(f'precision_at_{threshold}', 0) for threshold in thresholds]
        ax1.plot(thresholds, values, marker='o', linewidth=2, label=tool, color=tool_colors.get(tool, None))
    
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=20)
    ax1.set_xlabel('Rank Threshold')
    ax1.set_ylabel('Precision')
    ax1.set_xticks(thresholds)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Precision curves for cancer samples
    ax2 = fig4.add_subplot(2, 2, 2)
    for tool in tools_cancer:
        values = [benchmarker_cancer.metrics[tool]['accuracy'].get(f'precision_at_{threshold}', 0) for threshold in thresholds]
        ax2.plot(thresholds, values, marker='o', linewidth=2, label=tool, color=tool_colors.get(tool, None))
    
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=20)
    ax2.set_xlabel('Rank Threshold')
    ax2.set_ylabel('Precision')
    ax2.set_xticks(thresholds)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Mean and median ranks for all samples
    ax3 = fig4.add_subplot(2, 2, 3)
    mean_ranks = [benchmarker_all.metrics[tool]['ranking'].get('mean_rank', 0) for tool in tools_all]
    median_ranks = [benchmarker_all.metrics[tool]['ranking'].get('median_rank', 0) for tool in tools_all]
    x = np.arange(len(tools_all))
    width = 0.35
    ax3.bar(x - width/2, mean_ranks, width, label='Mean Rank')
    ax3.bar(x + width/2, median_ranks, width, label='Median Rank')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=20)
    ax3.set_xlabel('Tool')
    ax3.set_ylabel('Rank')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tools_all)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Panel D: Mean and median ranks for cancer samples
    ax4 = fig4.add_subplot(2, 2, 4)
    mean_ranks_cancer = [benchmarker_cancer.metrics[tool]['ranking'].get('mean_rank', 0) for tool in tools_cancer]
    median_ranks_cancer = [benchmarker_cancer.metrics[tool]['ranking'].get('median_rank', 0) for tool in tools_cancer]
    x_cancer = np.arange(len(tools_cancer))
    ax4.bar(x_cancer - width/2, mean_ranks_cancer, width, label='Mean Rank')
    ax4.bar(x_cancer + width/2, median_ranks_cancer, width, label='Median Rank')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=20)
    ax4.set_xlabel('Tool')
    ax4.set_ylabel('Rank')
    ax4.set_xticks(x_cancer)
    ax4.set_xticklabels(tools_cancer)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between subplots
    plt.savefig(os.path.join(output_dir, 'Figure4_precision_ranks.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Statistical tests
    print("Generating Figure 5: Statistical validation...")
    fig5 = plt.figure(figsize=(20, 15))
    
    # Panel A: Bootstrap CIs for all samples
    ax1 = fig5.add_subplot(3, 2, 1)
    metrics_list = ['top_rank_percentage', 'top_5_percentage', 'top_10_percentage']
    labels = ['Top-1', 'Top-5', 'Top-10']
    x_pos = np.arange(len(labels))
    
    for i, tool in enumerate(tools_all):
        if tool in benchmarker_all.statistical_tests['bootstrap_ci']:
            means, lowers, uppers = [], [], []
            for metric in metrics_list:
                ci_data = benchmarker_all.statistical_tests['bootstrap_ci'][tool][metric]
                means.append(ci_data['mean'])
                lowers.append(ci_data['lower'])
                uppers.append(ci_data['upper'])
            
            offset = (i - len(tools_all)/2) * 0.1
            yerr = [np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)]
            ax1.errorbar(x_pos + offset, means, yerr=yerr, fmt='o', capsize=5, 
                        label=tool, color=tool_colors.get(tool, None))
    
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=20)
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Bootstrap CIs for cancer samples
    ax2 = fig5.add_subplot(3, 2, 2)
    for i, tool in enumerate(tools_cancer):
        if tool in benchmarker_cancer.statistical_tests['bootstrap_ci']:
            means, lowers, uppers = [], [], []
            for metric in metrics_list:
                ci_data = benchmarker_cancer.statistical_tests['bootstrap_ci'][tool][metric]
                means.append(ci_data['mean'])
                lowers.append(ci_data['lower'])
                uppers.append(ci_data['upper'])
            
            offset = (i - len(tools_cancer)/2) * 0.1
            yerr = [np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)]
            ax2.errorbar(x_pos + offset, means, yerr=yerr, fmt='o', capsize=5, 
                        label=tool, color=tool_colors.get(tool, None))
    
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=20)
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Nemenyi test for all samples
    ax3 = fig5.add_subplot(3, 2, 3)
    if 'nemenyi_test' in benchmarker_all.statistical_tests and 'tool_pairs' in benchmarker_all.statistical_tests['nemenyi_test']:
        p_matrix = np.ones((len(tools_all), len(tools_all)))
        for (tool1, tool2), data in benchmarker_all.statistical_tests['nemenyi_test']['tool_pairs'].items():
            i, j = tools_all.index(tool1), tools_all.index(tool2)
            p_matrix[i, j] = p_matrix[j, i] = data['p_value']
        
        mask = np.triu(np.ones_like(p_matrix, dtype=bool))
        sns.heatmap(p_matrix, mask=mask, cmap='YlOrRd_r', vmin=0, vmax=0.1,
                   annot=True, fmt='.3f', cbar_kws={'label': 'p-value'}, ax=ax3)
        ax3.set_title('C', loc='left', fontweight='bold', fontsize=20)
        ax3.set_xticklabels(tools_all, rotation=45)
        ax3.set_yticklabels(tools_all)
    
    # Panel D: Nemenyi test for cancer samples
    ax4 = fig5.add_subplot(3, 2, 4)
    if 'nemenyi_test' in benchmarker_cancer.statistical_tests and 'tool_pairs' in benchmarker_cancer.statistical_tests['nemenyi_test']:
        p_matrix_cancer = np.ones((len(tools_cancer), len(tools_cancer)))
        for (tool1, tool2), data in benchmarker_cancer.statistical_tests['nemenyi_test']['tool_pairs'].items():
            i, j = tools_cancer.index(tool1), tools_cancer.index(tool2)
            p_matrix_cancer[i, j] = p_matrix_cancer[j, i] = data['p_value']
        
        mask_cancer = np.triu(np.ones_like(p_matrix_cancer, dtype=bool))
        sns.heatmap(p_matrix_cancer, mask=mask_cancer, cmap='YlOrRd_r', vmin=0, vmax=0.1,
                   annot=True, fmt='.3f', cbar_kws={'label': 'p-value'}, ax=ax4)
        ax4.set_title('D', loc='left', fontweight='bold', fontsize=20)
        ax4.set_xticklabels(tools_cancer, rotation=45)
        ax4.set_yticklabels(tools_cancer)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between subplots
    plt.savefig(os.path.join(output_dir, 'Figure5_statistical_tests.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined panel figures saved to {output_dir}/")


def main():
    """Main function to run the benchmark analysis with panel generation."""
    base_dir = r"C:\Users\z5537966\OneDrive - UNSW\Desktop\new data\test\common_samples_only\Franklin\manual\final_combined_data\Testis\limited_samples"
    unified_acmg_path = os.path.join(base_dir, "ACMG_185.csv")
    lirical_folder_path = base_dir
    manual_annotations_path = os.path.join(base_dir, "hgnc_standardized_matched_manual_annotations _185.xlsx")
    
    combined_output_dir = os.path.join(base_dir, "publication_figures")
    
    # Define cancer samples first
    cancer_samples = [
        # Original cancer samples
        'PGERA2440', 'PGERA1112', 'PGERA2125', 'PGERA1788', 'G5500', 'G5620', 
        'G2001380', 'G2101361', 'G1800091', 'G1800228', 
        'G2200657', 'G1900091', 'G2000091', 'G1900228', 'G2000228', 'G2101380', 
        'G2201380', 'G2201360', 'G2301360', 'G2300657', 'G2400657', 'G6500', 
        'G7500', 'G7620', 'PGERA2112', 'PGERA3112', 'PGERA2788', 
        'PGERA3788', 'PGERA3125', 'PGERA4125', 'PGERA3440', 'PGERA4440'
    ]
    
    # Get all samples and exclude cancer samples for the first analysis
    import pandas as pd
    manual_data = pd.read_excel(manual_annotations_path)
    
    # Debug: Check what columns are available
    print("Available columns in manual annotations:")
    print(manual_data.columns.tolist())
    
    # Try to find the correct sample ID column
    sample_id_column = None
    possible_names = ['Sample id', 'sample_id', 'Sample_id', 'SampleID', 'Sample ID', 'ID', 'sample']
    
    for col_name in possible_names:
        if col_name in manual_data.columns:
            sample_id_column = col_name
            break
    
    if sample_id_column is None:
        print("Error: Could not find sample ID column. Please check the column names above.")
        return
    
    print(f"Using column: '{sample_id_column}' for sample IDs")
    
    all_sample_ids = manual_data[sample_id_column].unique().tolist()
    non_cancer_samples = [sample for sample in all_sample_ids if sample not in cancer_samples]
    
    print(f"Total samples: {len(all_sample_ids)}")
    print(f"Cancer samples: {len(cancer_samples)}")
    print(f"Non-cancer (Mendelian) samples: {len(non_cancer_samples)}")
    
    print("\n===== BENCHMARKING NON-CANCER SAMPLES (EXCLUDING CPSR AND CHARGER) =====")
    benchmarker_all = VariantToolBenchmarker(
        unified_acmg_path,
        lirical_folder_path,
        manual_annotations_path,
        exclude_tools=['charger', 'cpsr', 'CPSR'],
        sample_subset=non_cancer_samples,  # Only non-cancer samples
        analysis_type='mendelian'
    )
    benchmarker_all.load_data()
    benchmarker_all.calculate_metrics()
    benchmarker_all.perform_statistical_tests()
    benchmarker_all.calculate_auc()
    
    print("\n\n===== BENCHMARKING CANCER SAMPLES (INCLUDING CPSR) =====")
    benchmarker_cancer = VariantToolBenchmarker(
        unified_acmg_path,
        lirical_folder_path,
        manual_annotations_path,
        exclude_tools=['charger'], 
        sample_subset=cancer_samples,
        analysis_type='cancer'
    )
    benchmarker_cancer.load_data()
    benchmarker_cancer.calculate_metrics()
    benchmarker_cancer.perform_statistical_tests()
    benchmarker_cancer.calculate_auc()
    
    # Generate combined panel figures
    print("\n===== GENERATING COMBINED PANEL FIGURES =====")
    generate_combined_panels(benchmarker_all, benchmarker_cancer, combined_output_dir)
    
    # Generate reports
    mendelian_report = os.path.join(combined_output_dir, "mendelian_samples_report.txt")
    cancer_report = os.path.join(combined_output_dir, "cancer_samples_report.txt")
    benchmarker_all.generate_report(mendelian_report)
    benchmarker_cancer.generate_report(cancer_report)
    
    print(f"\nAnalysis complete! Publication-ready figures saved to: {combined_output_dir}")

if __name__ == "__main__":
    main()
