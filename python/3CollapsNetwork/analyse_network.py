import numpy as np
from tqdm import tqdm


def collaps_network(network_T_full,nn_full,all_KW_full):
    # ToDo: collaps_synonyms.py - combining synonyms

    degree = np.count_nonzero(nn_full, 1)

    non_zero_index = degree.nonzero()[0]
    network_T_s2 = network_T_full[non_zero_index, :][:, non_zero_index]
    nn_s2 = nn_full[non_zero_index, :][:, non_zero_index]
    all_KW_s2 = [all_KW_full[i] for i in non_zero_index]

    # degree=sum(np.heaviside(nn_full,0))
    # orig_size = len(network_T_full)
    # Remove keywords that have never been used
    # network_T_s1=network_T_full[0,:]
    # nn_s1=nn_full[0,:]
    # orig_size_pbar = tqdm(range(orig_size))
    # for ii in orig_size_pbar:
    #     orig_size_pbar.set_description(f'Vertical (collapsed size: {len(network_T_s1)})')
    #     if degree[ii]>0:
    #         network_T_s1=np.vstack([network_T_s1, network_T_full[ii,:]])
    #         nn_s1=np.vstack([nn_s1, nn_full[ii,:]])
    # network_T_s1=network_T_s1[1:,:]
    # nn_s1=nn_s1[1:,:]
    #
    # network_T_s1=network_T_s1.transpose()
    # nn_s1=nn_s1.transpose()
    #
    # network_T_s2=network_T_s1[0,:]
    # nn_s2=nn_s1[0,:]
    # all_KW_s2=[]
    # orig_size_pbar = tqdm(range(orig_size))
    # for ii in orig_size_pbar:
    #     orig_size_pbar.set_description(f"Horizontal ({len(network_T_s2)})")
    #     if degree[ii]>0:
    #         network_T_s2=np.vstack([network_T_s2, network_T_s1[ii,:]])
    #         nn_s2=np.vstack([nn_s2, nn_s1[ii,:]])
    #         all_KW_s2.append(all_KW_full[ii])
    # network_T_s2=network_T_s2[1:,:]
    # nn_s2=nn_s2[1:,:]
    
    return network_T_s2, nn_s2, all_KW_s2
