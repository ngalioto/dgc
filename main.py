import numpy as np
import scanpy as sc
import pandas as pd
from scipy.optimize import nnls
import argparse
from utils import *

class DGC:

    def __init__(
        self,
        gene_list : list = None
    ):
        
        self.A = None
        self.xdim = None
        self.get_B_matrix(gene_list)

    def get_B_matrix(
        self, 
        gene_list=None
    ):
        """
        Instantiates the B matrix for DGC as an AnnData object.
        Rows are genes and columns are transcription factors.
        Entry b_ij = 1 if TF j can influence gene i (based on data from either HuRI or STRING). Otherwise b_ij = 0. 
        """

        self.B = sc.read_h5ad('data/B_matrix_2015.h5ad')
        gene_access = pd.read_csv('data/gene_accessibility.csv', index_col=0)

        # subset to genes in B matrix with same order
        mask = (gene_access.loc[self.B.obs_names]).values

        self.B.X = self.B.X * mask
        self.B = self.B[gene_list]

        self.B = map_genes_to_TADs(self.B, axis='obs')


    def build_A_matrices(
        self,
        adata : sc.AnnData
    ):
        
        """
        Computes array of the time-varying A matrices for the DGC model and assigns it to self.A
        self.A = [A_{n-1}, A_{n-1}A_{n-2}, ..., A_{n-1}...A_1]
        """

        num_data = adata.shape[0]
        xdim = adata.X.shape[1]

        I_tad = np.identity(xdim)
        A = np.zeros((num_data - 1, xdim, xdim))

        for ii, tt in enumerate(range(num_data - 1, 0, -1)):
            # Compute A for time (tt-1)
            num = np.outer(adata.X[tt] - adata.X[tt-1], adata.X[tt-1])
            den = np.dot(adata.X[tt-1], adata.X[tt-1]) 

            if ii == 0:
                A[ii] = (num / den) + I_tad
            else:
                A[ii] = A[ii-1] @ ((num / den) + I_tad)

        self.A = A
        self.xdim = xdim
    

    def estimate_tfs_constant(
        self,
        initial : np.ndarray,
        target : np.ndarray,
        recipe_list : list
    ):

        """
        Estimates the transcription factors according to DGC

        TODO: double check this optimization
        """

        Cbar = np.eye(self.xdim) + np.sum(self.A[1:], axis=0)
        b = target - self.A[-1] @ initial

        distances = {}
        for tfs in recipe_list:
            # select columns corresponding to the TFs
            B = self.B.X[:, self.B.var_names.isin(tfs)]

            # solve the non-negative least squares problem
            u, d = nnls(Cbar @ B, b)

            # save the results to the dictionary
            distances[tfs] = {'d': d, 'u': u}

        return distances


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--initial', type=str, default='fibroblast')
    argparser.add_argument('--target', type=str, default='myotube')
    argparser.add_argument('--recipe_len', type=int, default=1)
    argparser.add_argument('--target_expression', type=float, default=10.)
    argparser.add_argument('--fold_change', type=float, default=4.65)
    args = argparser.parse_args()

    if args.initial != 'fibroblast':
        raise ValueError(f'In the current version, initial cell must be fibroblast but got {args.initial}')

    cell_dict = {
        'esc' : 'ESC (ENCODE GSE23316)',
        'myotube' : 'Myotube (ENCODE GSE52529)'
    }

    adata = sc.read_h5ad('data/A_matrix_2015.h5ad')
    cell_profiles = sc.read_h5ad('data/cell_targets.h5ad')

    target = cell_profiles[cell_profiles.obs.index == cell_dict[args.target]].copy()

    adata, target = preprocess_data(adata, target)
    dgc = DGC(gene_list=adata.var_names.to_list())

    all_tfs = np.array(dgc.B.var_names.to_list())
    candidate_tfs = filter_tfs(
        initial = adata[0], 
        target = target, 
        target_expression = args.target_expression, 
        fold_change = args.fold_change, 
        tf_list = all_tfs
    )
    recipe_list = generate_recipes(candidate_tfs, args.recipe_len)

    adata = map_genes_to_TADs(adata)
    target = map_genes_to_TADs(target)

    d0 = np.linalg.norm(adata.X[-1] - target.X[0], ord=2)

    dgc.build_A_matrices(adata)

    sol = dgc.estimate_tfs_constant(
        initial = adata.X[0], 
        target = target.X[0],
        recipe_list = recipe_list
    )

    sorted_scores = {', '.join(k): d0 - v['d'] for k, v in sorted(sol.items(), key=lambda item: item[1]['d'])}

    print_scores(sorted_scores)