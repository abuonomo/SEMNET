import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def distance_between_nodes(mat):
    G=nx.from_numpy_matrix(np.matrix(mat))
    dist=np.zeros(mat.shape)
    
    for ii in range(len(mat)-1):
        for jj in range(ii+1,len(mat)):
            try:
                curr_dist=nx.shortest_path_length(G, ii, jj) 
            except:
                curr_dist=10 # if not connected: some large number, in paper it was 2*max(dist)
            dist[ii][jj]=curr_dist
            dist[jj][ii]=curr_dist
            
    return dist


def distance_between_nodes_weighted(mat):
    G=nx.from_numpy_matrix(np.matrix(mat))
    dist=np.zeros(mat.shape)
    
    for ii in range(len(mat)-1):
        for jj in range(ii+1,len(mat)):
            try:
                curr_dist=nx.dijkstra_path_length(G, ii, jj) 
            except:
                curr_dist=10 # some large number, in paper it was 2*max(dist)
                    
            dist[ii][jj]=curr_dist
            dist[jj][ii]=curr_dist
            
    return dist


def cos_similarity(mtx):    
    single_net=np.heaviside(mtx, 0.0)
    cos_sim=dot(single_net, single_net)/(norm(single_net)*norm(single_net))
    return cos_sim


def path_N(mtx,N):
    mult_mat=mtx
    for ii in range(1,N):
        mult_mat=dot(mult_mat, mtx)
    
    path_N_mtx=mult_mat/(norm(mtx)*norm(mtx))
    return path_N_mtx


def log_or_pbar(msg, pbar):
    if pbar is not None:
        pbar.set_description(msg)
    else:
        LOG.info(msg)


def calculate_all_network_properties_per_year(single_net,single_netYM1,single_netYM2,evolving_nums, pbar=None):
    # we create 17 matrices of properties
    # numsA, numsB, degA, degB (those vectors need to be stored only once)
    # cosS, 
    # path2Y,path2YM1,path2YM2,path3Y,path3YM1,path3YM2,path4Y,path4YM1,path4YM2
    # distance, weighted_distance_1, weighted_distance_2
    # Those properties have been used based on heuristic tests, probably there is a lot of room for improvements

    log_or_pbar("numsA, numsB, degA, degB", pbar)
    all_properties=[]
    all_properties.append(np.array(evolving_nums))
    
    degrees=np.count_nonzero(single_net,axis=0)
    all_properties.append(degrees/max(degrees))

    log_or_pbar("cosS", pbar)
    cos_sim = cos_similarity(single_net)
    all_properties.append(cos_sim)

    log_or_pbar("paths", pbar)
    for curr_mat in [single_net,single_netYM1,single_netYM2]:
        for ii in range(2,5):
            all_properties.append(path_N(curr_mat,ii))

    uncon_nodes_num = 10.
    log_or_pbar("distance", pbar)
    dist_mtx = shortest_path(single_net, unweighted=True)
    dist_mtx[np.isinf(dist_mtx)] = uncon_nodes_num  # unconnected nodes set to a large number
    all_properties.append(dist_mtx)

    log_or_pbar("weighted distance", pbar)
    epsilon=10**(-4)  # simple way to avoid runtime warning, div by zero (which has no consequence for result)
    out_degrees = np.outer(degrees, degrees)
    out_degrees_sqrt = np.sqrt(out_degrees)
    tmp_sing = single_net + epsilon
    w_mtx1 = out_degrees / tmp_sing
    w_mtx2 = out_degrees_sqrt / tmp_sing

    dist_mtx1 = shortest_path(w_mtx1, unweighted=False)
    dist_mtx1[np.isinf(dist_mtx1)] = uncon_nodes_num
    dist_mtx2 = shortest_path(w_mtx2, unweighted=False)
    dist_mtx2[np.isinf(dist_mtx2)] = uncon_nodes_num

    all_properties.append(dist_mtx1)
    all_properties.append(dist_mtx2)   
    
    return all_properties



def calculate_all_network_properties(evolving_nets,evolving_nums):
    all_properties_years=[]
    evolving_nets_pbar = tqdm(range(2,len(evolving_nets)), total=len(evolving_nets)-2)
    print('calculate_all_network_properties')
    for ii in evolving_nets_pbar:
        current_all_properties=calculate_all_network_properties_per_year(
            evolving_nets[ii],
            evolving_nets[ii-1],
            evolving_nets[ii-2],
            evolving_nums[ii],
            pbar=evolving_nets_pbar,
        )
        all_properties_years.append(current_all_properties)
    
    return all_properties_years
