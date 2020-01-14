import sys
import logging
import joblib

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

sys.path.append("1aCreateFullArticleData")
sys.path.append("1bCreateFullArticleData_APS")
sys.path.append("2CreateNetwork")
sys.path.append("3CollapsNetwork")
sys.path.append("4CalculateAncientNetworks")

from create_full_article_data import create_full_data_arxiv
from create_full_article_data_APS import create_full_data_APS
from create_network import create_network
from analyse_network import collaps_network
from create_ancient_network import create_ancient_network
from calc_all_values_optimized import calc_all_values_optimized

# full_data_arxiv = create_full_data_arxiv()
# full_data_APS = create_full_data_APS()
# all_papers = full_data_arxiv + full_data_APS
#
# all_networks_full, nn_full, all_KW_full, all_years = create_network(all_papers)
# all_networks, nn, all_KW = collaps_network(all_networks_full, nn_full, all_KW_full)
# # _ = create_ancient_network(all_networks, nn, all_KW)
# joblib.dump([all_years, all_networks, nn, all_KW], 'state_vars.jbl')
all_years, all_networks, nn, all_KW = joblib.load('state_vars.jbl')
_ = calc_all_values_optimized(all_years, all_networks, nn, all_KW)

# TODO:
# 4CalculateAncientNetworks) Create ancient Networks
# The data exists in the network_T array. There, the year of the connection is
# thus one can straight forward make semantic networks for each year.

# 5PrepareNNData)Prepare the network theoretical properties, which are descibed in the SI of the paper (https://arxiv.org/abs/1906.06843)

# 6TrainNN) Train network, we will use PyTorch

# 7bCalcROC) Evaluate the Quality of the predictions using ROC and AUC
