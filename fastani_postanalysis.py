#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: get_redundant_genomes.py
Description: This script take fastani output and return list of non-redundat genomes
             with their respective redundant genomes.
Author: Guillermo Uceda-Campos
Date: July 2023
Version: 1.0
Requirements:
- pandas, numpy, markov_clustering, and scipy
Example of use:
- get_redundant_genomes.py fastani.output 0.99
"""

import os
import argparse
import numpy as np
import pandas as pd
import markov_clustering as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix
from argparse import RawTextHelpFormatter

def fastanitab_to_dataframe(fastani_output):
    """
    Description: This function builds a dataframe from fastani output
    Parameters:
    - parameter1: file of the fastani output
    Returns:
    - Dataframe containing query, reference, identity, and coverage.
    """
    df_fastani = pd.read_table(fastani_output, names=['q','r','id','frag_match','frag_total'])
    df_fastani.q = df_fastani.q.str.split('/').str[-1].str.split('_').str[:2].str.join('_')
    df_fastani.r = df_fastani.r.str.split('/').str[-1].str.split('_').str[:2].str.join('_')
    # Adding column coverage
    df_fastani['co'] = df_fastani.frag_match/df_fastani.frag_total*100
    return df_fastani

def get_triangular_matrix(df_fastani, id_or_cov):
    """
    Description: This function gets a triangular matrix where each cell corresponds to 
    the unique value (mean) of ANI comparisons, e.g. 
    if ANI-ab = 99.5 and ANI-ba = 99.9, the unique value in the
    triangular matrix (shown in the upper part) will be ANI-ab = 99.7,
    and the other cells will be filled with 0.
    Parameters:
    - parameter1: a dataframe containing the matrix of fastani output (all vs all)
    Returns:
    - Triangular matrix containing the ANI means of each genome comparison.
    """

    # Making a matrix with the ANI values (bi-directional: A->B and B->A)
    M_nodes = sorted(pd.unique(df_fastani[['q', 'r']].values.ravel()))
    M = pd.DataFrame(index=M_nodes, columns=M_nodes, dtype='float')
    for i, row in df_fastani.iterrows():
        M.at[row['q'], row['r']] = row[id_or_cov]
    M.fillna(0, inplace=True)

    # Getting the upper matrix
    M_upper = pd.DataFrame(np.triu(M), columns=M.columns, index=M.index)
    # Getting the lower transporter matrix
    M_lowerT = pd.DataFrame(np.tril(M), columns=M.columns, index=M.index).T

    # M_triangular matrix contains the average of A->B and B->A
    M_triangular = M_upper + M_lowerT
    M_triangular = M_triangular/2

    return M_triangular

def make_mcl_analysis(M_triangular, id_or_co_cut):
    """
    Description: This function takes a triangular matrix and fills with zeros
    the cells with values less than id_cut parameter.
    The purpose of this step is to get a sparse matrix and apply MCL.
    Parameters:
    - parameter1: Triangular matrix containing the ANI means of each genome comparison.
    - parameter2: Minimum percentage of identity among the redundant group of genomes.
    Returns:
    - List of redundant (in the same row) and non-redundant (each line) genomes.
    """
    M_triangular_i = M_triangular.where(M_triangular >= id_or_co_cut, 0)
    M_sparse = csr_matrix(M_triangular_i.values)
    mcl_results = mc.run_mcl(M_sparse, inflation=2)
    mcl_clusters = mc.get_clusters(mcl_results)
    
    redundant_genome_groups = []
    names_sorted_by_groups = []
    for c in mcl_clusters:
        c_names = []
        for index in c:
            name = M_triangular.columns.tolist()[index]
            names_sorted_by_groups.append(name)
            c_names.append(name)
        redundant_genome_groups.append(c_names)

    # LIST (Dataframe) of redundant genomes
    redundant_genome_groups = pd.DataFrame(redundant_genome_groups)

    # MATRIX (Dataframe) of ANI-means sorted by mcl groups
    M_triang_mcl = M_triangular[names_sorted_by_groups].reindex(names_sorted_by_groups)
    # extra step in sortering ...
    M_triang_mcl_upper = pd.DataFrame(np.triu(M_triang_mcl), columns=M_triang_mcl.columns, index=M_triang_mcl.index)
    M_triang_mcl_lowerT = pd.DataFrame(np.tril(M_triang_mcl), columns=M_triang_mcl.columns, index=M_triang_mcl.index).T

    M_triang_mcl_upper[M_triang_mcl_upper == 0] = M_triang_mcl_lowerT
    M_trinagular_final = M_triang_mcl_upper

    return redundant_genome_groups, M_trinagular_final

def make_plot(M_triangular, plot_filename, vmin):
    """
    Description: Make a plot of the matrices
    Parameters:
    - M_triangular: Triangular matrix to plot
    - plot_filename: Filename to save the plot
    - vmin: Minimum value for color scale
    """

    # Labels
    xlabs = M_triangular.columns
    ylabs = M_triangular.index
    # Heat map
    fig, ax = plt.subplots(figsize=(xlabs.shape[0]/5, ylabs.shape[0]/5))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white','yellow', 'red'], N=256)
    im = ax.imshow(M_triangular.values, cmap=cmap,interpolation='nearest', aspect='auto', vmin=vmin)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    fig.colorbar(im, ax=ax, cax=cax)

    # Add the labels
    ax.set_xticks(np.arange(len(xlabs)))
    ax.set_xticklabels(xlabs, rotation=90, ha='right')
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_yticks(np.arange(len(ylabs)))
    ax.set_yticklabels(ylabs)
    ax.yaxis.set_tick_params(labelsize=6)
    # Save the plot to file
    fig.savefig(plot_filename, bbox_inches='tight')

def main(fastani_tab, out=False, identity=99.9, coverage=99.9, method='m1', matrix=False, plot=False):
    """
    Description: This function is the entry point of the script and performs the main analysis.
    Parameters:
    - fastani_tab: Fastani output file
    - out: Text file with groups of redundant genomes by line (default: $fani.tab)
    - identity: Minimum percentage of identity among the group of redundant genomes (default: 99.9)
    - coverage: Minimum percentage of coverage among the group of redundant genomes (not yet available)
    - method: m1: redundant genomes based on the --identity (default)
              m2: redundant genomes based on the --identity and --coverage (not yet available)
    - matrix: Make matrix if True (default: False)
    - plot: Make plot if True (default: False)
    """

    # STEP 0 : Defining basename to outputs and identifying directory
    fastani_filename = os.path.basename(fastani_tab) # basename of fastani table (input)
    basename_out = fastani_filename # basename of outputs (default)
    if out:
        basename_out = os.path.basename(out)
    fastani_dirname = os.path.dirname(fastani_tab) # dirname of fastani output
    
    # STEP 1: Dataframe from fastani output
    df_fastani = fastanitab_to_dataframe(fastani_tab)

    # STEP 2 : Getting the triangular matrix
    M_triangular_id = get_triangular_matrix(df_fastani,'id')
    M_triangular_co = get_triangular_matrix(df_fastani,'co')

    # STEP 3 : Getting the list of redundant and non-redundant genomes, and matrix with all genomes sorted by mcl clustering
    redundant_genome_groups_id, ani_mean_matrix_id = make_mcl_analysis(M_triangular_id, identity)
    redundant_genome_groups_co, M_trinagular_final_co = make_mcl_analysis(M_triangular_co, coverage)

    # STEP 4 : Wrinting list of non-redundant genomes (text file)
    if method == None:
        method = 'm1' # Deafaut
    
    if method == 'm1':
        redundant_genome_filename = os.path.join(fastani_dirname, basename_out)
        redundant_genome_groups_id.to_csv(redundant_genome_filename+'.id.tab', sep='\t', header=False, index=False)
    elif method == 'm2':
        pass # Future implementation
    else:
        print('incorrect method')

    # STEP 5: Writting matrix if True
    if matrix:
        M_trinagular_filename = os.path.join(fastani_dirname, basename_out)
        ani_mean_matrix_id.to_csv(M_trinagular_filename+'.id.matrix', sep='\t', header=True, index=True)
        M_trinagular_final_co.to_csv(M_trinagular_filename+'.co.matrix', sep='\t', header=True, index=True)

    # STEP 6: Ploting matrix if True
    if plot:
        plot_filename = os.path.join(fastani_dirname, basename_out)
        make_plot(M_triangular_id, plot_filename+'.id.png', vmin=99.5)
        make_plot(M_triangular_co, plot_filename+'.co.png', vmin=99.0)

    # STEP 7 : Printing brief report in terminal
    print(basename_out,' -> Initital genomes: ', ani_mean_matrix_id.shape[0], ' Non-redundant genomes',redundant_genome_groups_id.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''This script generates:\n \
        1. List of redundant genomes based on the available method\n \
        2. Triangular matrix of percent identity and coverage between genomes (if --matrix True)\n \
        3. Plot of the triangular matrices (if --plot True)''',
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('--fani', type=str, help='Fastani output file')
    parser.add_argument('--out', type=str, help='Text file with groups of redundant genomes by line (default: $fani.tab)')
    parser.add_argument('--identity', type=float, help='Minimum percentage of identiy among the group of redundant genomes (default: 99.9)')
    parser.add_argument('--coverage', type=float, help='Minimum percentage of coverage among the group of redundant genomes (not yet available)')
    parser.add_argument('--method', type=str, help='m1: redundant genomes based in the --identity (default)\nm2:redundant genomes based in the --identity and --coverage (not yet available)')
    parser.add_argument('--matrix', type=bool, help='Make matrix if True (default: False)')
    parser.add_argument('--plot', type=bool, help='Make plot if True (default: False)')
    
    args = parser.parse_args()
    fastani_tab = args.fani
    main(fastani_tab, out=args.out, identity=args.identity, coverage=args.coverage, method=args.method, matrix=args.matrix, plot=args.plot)