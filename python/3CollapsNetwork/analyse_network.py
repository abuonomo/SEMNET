import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def collaps_network(all_networks, nn_full, all_KW_full):
    # ToDo: collaps_synonyms.py - combining synonyms

    LOG.info('Collapsing networks to only use keywords which appear in at least one.')
    degree = np.count_nonzero(nn_full, 1)
    non_zero_index = degree.nonzero()[0]
    all_networks_s2 = {}
    year_network_pbar = tqdm(all_networks.items())
    for year, network in year_network_pbar:
        year_network_pbar.set_description(str(year))
        all_networks_s2[year] = network[non_zero_index, :][:, non_zero_index]
    nn_s2 = nn_full[non_zero_index, :][:, non_zero_index]
    all_KW_s2 = [all_KW_full[i] for i in non_zero_index]

    return all_networks_s2, nn_s2, all_KW_s2
