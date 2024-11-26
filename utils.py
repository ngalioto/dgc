import numpy as np
import scanpy as sc
import itertools
from typing import Tuple

def filter_tfs(
    initial : sc.AnnData,
    target : sc.AnnData,
    target_expression : float,
    fold_change : float,
    tf_list : np.ndarray
) -> np.ndarray:

    """
    Filters the transcription factors based on the target expression and fold change

    TODO: handle zeros in initial and target
    """
    
    initial = initial[:, np.isin(initial.var_names, tf_list)]
    target = target[:, np.isin(target.var_names, tf_list)]

    mask = np.logical_and(np.isin(tf_list, target.var_names), np.isin(tf_list, initial.var_names))

    expression_condition = target.X >= target_expression
    fold_change_condition = np.log2(target.X / initial.X) >= fold_change

    return tf_list[mask][np.logical_and(expression_condition, fold_change_condition).squeeze()]

def generate_recipes(
    tf_list : np.ndarray,
    n : int
) -> list[str]:

    """
    Generates all possible combinations of transcription factors
    """

    if n > 3 or n < 1:
        raise ValueError(f'The number of transcription factors must be 1, 2, or 3. Received {n=}')

    recipes = list(itertools.combinations(tf_list, 1))
    for ii in range(2, n+1):
        recipes.extend(itertools.combinations(tf_list, ii))
    return recipes

def preprocess_data(
    adata : sc.AnnData,
    target : sc.AnnData
) -> Tuple[sc.AnnData, sc.AnnData]:

    """
    Preprocesses the data for the DGC model
    """

    # load in steady-state value, ($\\bar{x}$ in the paper)
    steady_state = sc.read_h5ad('data/fibroblast_ss.h5ad')

    # Set microRNA genes to zero
    gene_idx_to_zero = np.arange(11165, 12324)
    adata.X[:, gene_idx_to_zero] = 0.
    target.X[:, gene_idx_to_zero] = 0.

    # Collect the set of genes that do not have expression values
    nan_genes = set(target.var_names[np.isnan(target.X).any(axis=0)])
    nan_genes.update(steady_state.var_names[np.isnan(steady_state.X).any(axis=0)])
    nan_genes.update(adata.var_names[np.isnan(adata.X).any(axis=0)])

    # Set microRNA genes to zero
    steady_state.X[:, gene_idx_to_zero] = 0.

    # Remove genes that do not have expression values
    target = target[:, ~target.var_names.isin(nan_genes)].copy()
    steady_state = steady_state[:, ~steady_state.var_names.isin(nan_genes)]
    adata  = adata[:, ~adata.var_names.isin(nan_genes)].copy()

    # center data
    adata.X = adata.X - steady_state.X
    target.X = target.X - steady_state.X

    return adata, target


def map_genes_to_TADs(
    adata : sc.AnnData,
    axis : str = 'var',
    func : str = 'sum'
) -> sc.AnnData:
    
    # Map the vectors to TAD space
    adata = sc.get.aggregate(adata, by='TAD', func=func, axis=axis)
    adata.X = adata.layers['sum']
    
    return adata

def print_scores(
    sorted_scores : dict[str, float]
) -> None:

    """
    Prints the scores from the optimization
    """

    max_recipe_len = max(len(k) for k in sorted_scores.keys())
    print(f"{'Recipe':<{max_recipe_len}}  | {'Score':>8}")

    # Print a separator line for clarity
    print("-" * (max_recipe_len + 3 + 10))
    for recipe, score in sorted_scores.items():
        print(f"{recipe:<{max_recipe_len}}  | {score:>8.2f}")