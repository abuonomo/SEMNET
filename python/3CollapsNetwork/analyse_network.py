import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DEFAULT_SYN_LOC = Path(__file__).parent.absolute() / Path('SynonymList.lst')

def collaps_synonyms(network_T_full,nn_full,all_KW_full, synonym_list: Path = DEFAULT_SYN_LOC):
    all_syn=[[]]
    syn_count=0
    with open(synonym_list) as fp:
        line = fp.readline()
        while line:
            if line[0:-1]=='---':
                all_syn.append([])
                syn_count+=1
            else:
                all_syn[syn_count].append(line[0:-1])
            line = fp.readline()
    
    
    synonym_indices=[]
    for ii in range(len(all_syn)): # all classes of synonyms (e.g., ['bb84','bb84 protocol'])
        KW_idx=[]
        for jj in range(len(all_KW_full)): # all KWs
            for kk in range(len(all_syn[ii])): # over all synonym members, 'bb84','bb84 protocol' 
                if all_syn[ii][kk]==all_KW_full[jj]: # is any of the synonym members the KW?
                    KW_idx.append(jj)
        KW_idx.sort()
        synonym_indices.append(KW_idx)
    
    
    all_removed_kws=[]
    for ii in range(len(synonym_indices)):
        if len(synonym_indices[ii])>1:
            curr_syn_idx=synonym_indices[ii]
            prime_idx=curr_syn_idx[0]
            for jj in range(1,len(curr_syn_idx)):            
                all_removed_kws.append(curr_syn_idx[jj])
                for kk in range(len(network_T_full)):
                    network_T_full[prime_idx,kk]+=network_T_full[curr_syn_idx[jj],kk]
                    network_T_full[kk,prime_idx]+=network_T_full[curr_syn_idx[jj],kk]
    
                    nn_full[prime_idx,kk]+=nn_full[curr_syn_idx[jj],kk]
                    nn_full[kk,prime_idx]+=nn_full[kk,curr_syn_idx[jj]]
                    
    all_removed_kws.sort(reverse=True)
    
    # all_removed_kws are KWs that we will delete, because they are
    # synonyms with other words, and we store their information
    network_T_full=np.delete(network_T_full,all_removed_kws,axis=0)
    network_T_full=np.delete(network_T_full,all_removed_kws,axis=1)
    nn_full=np.delete(nn_full,all_removed_kws,axis=0)
    nn_full=np.delete(nn_full,all_removed_kws,axis=1)
    
    for ii in all_removed_kws:
        del all_KW_full[ii]
        
        
    return network_T_full,nn_full,all_KW_full
            

def collaps_network(network_T_full,nn_full,all_KW_full, synonym_list=DEFAULT_SYN_LOC):
    # Remove keywords that are synonyms, but keep their information
    network_T_full,nn_full,all_KW_full=collaps_synonyms(network_T_full,nn_full,all_KW_full, synonym_list)
    print('collaps_network - Finished collapsing synonyms')

    degree = np.count_nonzero(nn_full, 1)
    non_zero_inds = np.argwhere(degree > 0).flatten()
    LOG.info(f'Collapsing to {len(non_zero_inds)} keywords.')
    network_T_s2 = network_T_full[non_zero_inds, :][:, non_zero_inds]
    nn_s2 = nn_full[non_zero_inds, :][:, non_zero_inds]
    all_KW_s2 = [all_KW_full[i] for i in non_zero_inds]

    return network_T_s2, nn_s2, all_KW_s2