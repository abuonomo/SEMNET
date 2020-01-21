import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

DEFAULT_KEYWORD_LOC = Path(__file__).parent.absolute() / Path('keyword_list.lst')


def get_full_text(article):
    full_text = article[1] + ' ' + article[2]
    full_text = full_text.replace("'", "")
    full_text = full_text.replace("--", "-")
    full_text = full_text.replace("ö", "oe")
    full_text = full_text.replace("ä", "ae")
    full_text = full_text.replace("ü", "ue")
    full_text = full_text.replace('{\"o}', 'oe')
    full_text = full_text.replace('\\', '')
    full_text = full_text.lower()
    return full_text


def remove_brackets(full_text: str) -> str:
    """
    Remove brackets, for example: Clauser-Horne (CH) inequality, in a separate string.
    Thus have two abstracts that can be disaminated

    Args:
        full_text: text to remove brackets from

    Returns:
        text with removed brackets
    """
    if '(' in full_text:  # Remove brakets, for example: Clauser-Horne (CH) inequality, in a separate string. Thus have two abstracts that can be disaminated
        idx = full_text.find('(') + 1
        new_full_text = full_text[0:idx - 1]
        bracket_count = 1
        while idx < len(full_text):
            if bracket_count == 0:
                new_full_text += full_text[idx]

            if full_text[idx] == '(':
                bracket_count += 1
            if full_text[idx] == ')':
                bracket_count -= 1

            idx += 1

        full_text = full_text + " " + new_full_text.replace('  ', ' ')
    return full_text


def get_found_kw(full_text, sorted_KW_idx, all_KW):
    found_KW = []
    for kw_idx in sorted_KW_idx:
        if all_KW[kw_idx] in full_text:
            found_KW.append(kw_idx)
            full_text = full_text.replace(all_KW[kw_idx], '')
    found_KW = list(set(found_KW))
    return found_KW


def create_network(all_papers, keyword_list: Path = DEFAULT_KEYWORD_LOC, limit=1500):
    with open(keyword_list) as fp:
        all_KW = [line.strip() for line in fp.readlines()]
    if limit is not None:
        LOG.info(f'For debugging reasons, only {limit} KWs are used')
        all_KW = all_KW[0:limit]
    KW_length = [len(kwd) for kwd in all_KW]

    sorted_KW_idx=np.argsort(KW_length)[::-1] # indices of KWs from largest to smallest

    padding_str=''
    for ii in range(1000):
        padding_str=padding_str+' '

    LOG.info('Creating Network Template')
    network_T=np.frompyfunc(list, 0, 1)(np.empty((len(all_KW),len(all_KW)), dtype=object))
    nn=np.zeros((len(all_KW),len(all_KW)))
    cc_papers=0
    all_papers_pbar = tqdm(all_papers)
    for article in all_papers_pbar:
        # quasi-unique paper identifier:
        # this number between (0,1) will be added to the year in SemNet
        # such that one number carries both the year-info and the paper id in the for YYYY.IDIDID
        paper_id=random.random()
        
        cc_papers+=1
        full_text = get_full_text(article)
        full_text = remove_brackets(full_text)
        found_KW = get_found_kw(full_text, sorted_KW_idx, all_KW)

        for ii in range(len(found_KW)-1):
            for jj in range(ii+1,len(found_KW)):
                network_T[found_KW[ii],found_KW[jj]].append(int(article[0][0:4])+paper_id)
                network_T[found_KW[jj],found_KW[ii]].append(int(article[0][0:4])+paper_id)
                
                nn[found_KW[ii],found_KW[jj]]+=1
                nn[found_KW[jj],found_KW[ii]]+=1

    return network_T, nn, all_KW