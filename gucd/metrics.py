"""Evaluation of generated flows for one origin (Fig. 3, Extended Data Table 1).

For origin i with observed flows T_ij and generated flows That_ij over its destinations j:
  CPC    = 2 sum_j min(That_ij, T_ij) / (sum_j That_ij + sum_j T_ij)   (common part of commuters)
  Spearman correlation between That_ij and T_ij
  RMSE   = sqrt(mean_j (That_ij - T_ij)^2), in trips
  NRMSE  = RMSE / standard deviation of the origin's observed flows
A city's value averages these over the test origins with observed trips, and then over the five splits.
"""
import numpy as np
from scipy.stats import spearmanr


def cpc(t_hat, t):
    t_hat, t = np.asarray(t_hat, float), np.asarray(t, float)
    return float(2 * np.minimum(t_hat, t).sum() / (t_hat.sum() + t.sum()))


def spearman(t_hat, t):
    return float(spearmanr(t_hat, t).correlation)


def rmse(t_hat, t):
    return float(np.sqrt(np.mean((np.asarray(t_hat, float) - np.asarray(t, float)) ** 2)))


def nrmse(t_hat, t):
    sd = np.std(np.asarray(t, float))
    return rmse(t_hat, t) / sd if sd > 0 else float("nan")
