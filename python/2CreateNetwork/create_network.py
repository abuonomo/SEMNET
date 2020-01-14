import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import dok_matrix
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def create_network(all_papers):
    keyword_list = Path('2CreateNetwork/keyword_list.lst')
    with open(keyword_list) as fp:
        all_KW = [s.strip() for s in fp.readlines()]

    padding_str=''
    for ii in range(1000):
        padding_str=padding_str+' '

    line=[ [] for _ in range(5)]
    LOG.info('Creating Network Template')
    all_networks = {}
    network_T = np.empty((len(all_KW), len(all_KW)))
    network_T[:] = np.nan
    network_T = network_T.astype(object)
    nn=np.zeros((len(all_KW),len(all_KW)))
    cc_papers=0
    all_papers_pbar = tqdm(all_papers)
    for article in all_papers_pbar:
        if cc_papers%10==0:
            all_papers_pbar.set_description(f'{article[1][0:50]}...')

        cc_papers+=1

        full_text=article[1]+' '+article[2]
        full_text=full_text.replace("'","")
        full_text=full_text.replace("--","-")
        full_text=full_text.replace("ö","oe")
        full_text=full_text.replace("ä","ae")
        full_text=full_text.replace("ü","ue")
        full_text=full_text.replace('{\"o}','oe')
        full_text=full_text.replace('\\','')
        full_text=full_text.lower()

        # ToDo: one can also remove brackets
        # such as "Greenberger-Horne-Zeilinger (GHZ) state", GHZ
        # Thus have two abstracts that can be disaminated

        found_KW=[]
        kw_idx=0
        for kw in all_KW:
            if full_text.find(kw)>=0:
                found_KW.append(kw_idx)            
                #   ToDo: padd this part of the text. in that way, other KWs dont use part of this KW anymore (idea is: this KW is more specific than smaller ones)   
                #   full_text.replace(kw,padding_str[0:len(kw)]) 
                #   However for this the KWs should be sorted by size
            kw_idx+=1

        found_KW=list(set(found_KW))
        for ii in range(len(found_KW)-1):
            for jj in range(ii+1,len(found_KW)):
                year = int(article[0][0:4])
                if year not in all_networks:
                    all_networks[year] = dok_matrix((len(all_KW), len(all_KW)))
                all_networks[year][found_KW[ii],found_KW[jj]] = 1
                all_networks[year][found_KW[jj], found_KW[ii]] = 1

                nn[found_KW[ii],found_KW[jj]] += 1
                nn[found_KW[jj],found_KW[ii]] += 1

    all_years = np.array(list(all_networks.keys()))
    all_years.sort()
    LOG.info(f'Made network for years {all_years}.')

    return all_networks, nn, all_KW, all_years