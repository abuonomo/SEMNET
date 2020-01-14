import argparse
import logging

import scipy.sparse
import numpy as np
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_distance_inner(g):
    sparse_g = scipy.sparse.csr_matrix(g)
    dist_g = scipy.sparse.csgraph.johnson(sparse_g)
    # Get double the max length between connected nodes
    big_dist = 2 * dist_g[~np.isinf(dist_g)].max()
    dist_g[np.isinf(dist_g)] = big_dist
    return dist_g


def get_distance(g: np.array) -> np.array:
    """
    Get distance between nodes in graph g.

    Args:
        g: a symmetric array which represents a graph of connected terms

    Returns:
        an array of distances between nodes, with inf represented as double the max
        non-infinite distance between nodes
    """
    g = g.copy()
    g[g > 0] = 1
    dist_g = get_distance_inner(g)
    return dist_g


def get_alt_distance_1(nn_ancient, deg):
    nn_ancient = nn_ancient.copy()
    deg_a = np.tile(deg, (nn_ancient.shape[0], 1))
    deg_b = np.tile(deg, (nn_ancient.shape[0], 1))
    mm1 = ((deg_a * deg_b) ** (1/2)) / nn_ancient
    mm1[np.isinf(mm1)] = 0
    mm1 = mm1 * (1 - np.identity(mm1.shape[0]))
    dist_alt1 = get_distance_inner(mm1)
    return dist_alt1


def get_alt_distance_2(nn_ancient, deg):
    hash = nn_ancient.sum(axis=1)
    hash_a = np.tile(hash, (nn_ancient.shape[0], 1))
    mm2 = np.matmul(hash_a, hash_a.T) / nn_ancient
    mm2 = mm2 * (1 - np.identity(mm2.shape[0]))
    dist_alt_2 = get_distance_inner(mm2)
    return dist_alt_2


def get_paths(mm1, mm2):
    mat = np.matmul(mm1, mm2)
    deg_n = mat.sum(axis=1)
    norm_n = np.outer(deg_n, deg_n) ** (1/2)
    paths_2_ancient = mat / norm_n
    paths_2_ancient[np.isnan(paths_2_ancient)] = 0
    return paths_2_ancient, mat


def calc_all_values_optimized(all_years, all_networks, nn, all_KW):
    all_year_pbar = tqdm(enumerate(all_years[::-1]), total=len(all_years))
    for i, year in all_year_pbar:
        all_year_pbar.set_description(str(year))
        network = all_networks[year]
        all_KW = np.array(all_KW)
        if i == 0:  # limit by the first year in the series? Need enough occurrences?
            deg0 = np.array((network > 0).sum(axis=0))[0]
            all_kwds_lim = all_KW[deg0 > 0]
            # save limited set
        nn_ancient = np.array(network[deg0 > 0, :][:, deg0 > 0].todense())
        nums_ancient = nn[deg0 > 0]
        deg = (nn_ancient > 0).sum(axis=0)
        dist_ancient = get_distance(nn_ancient)
        dist_alt_1 = get_alt_distance_1(nn_ancient, deg)
        dist_alt_2 = get_alt_distance_2(nn_ancient, deg)
        paths_2_ancient, mat = get_paths(nn_ancient, nn_ancient)
        paths_3_ancient, mat = get_paths(mat, nn_ancient)
        paths_4_ancient, mat = get_paths(mat, nn_ancient)
        paths_5_ancient, mat = get_paths(mat, nn_ancient)
        cos_sim, mat = get_paths((nn_ancient > 0), (nn_ancient > 0))
    pass



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Say hello')
    # parser.add_argument('i', help='input txt file')
    # parser.add_argument('--feature', dest='feature', action='store_true')
    # parser.add_argument('--no-feature', dest='feature', action='store_false')
    # parser.set_defaults(feature=True)
    # args = parser.parse_args()
    # main(args.i, args.feature)
    main()
