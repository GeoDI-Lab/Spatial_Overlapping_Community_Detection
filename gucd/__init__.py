"""Weighted, overlapping community memberships of urban mobility networks.

  data         networks (A = OD + OD^T), observed pairs, published memberships
  node2vec     the node2vec embedding used as input
  model        the graph convolutional network that infers F, and its training
  conversions  F, its binary version and its hard partition as fitted flow models (L1)
  overlap      overlap count and overlap zone
  metrics      common part of commuters, Spearman correlation, RMSE and NRMSE of generated flows
"""
__version__ = "1.0.0"
