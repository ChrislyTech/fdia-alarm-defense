# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gaussian_kde

def calculate_threshold(statistics, alpha=0.995):
    """
    Calculate the threshold using Kernel Density Estimation (KDE).
    :param statistics: List or array of statistics (e.g., T2 or SPE values).
    :param alpha: Confidence level (default is 0.995).
    :return: Threshold value.
    """
    kde = gaussian_kde(statistics)
    x = np.linspace(min(statistics), max(statistics), 1000)
    pdf = kde(x)
    cdf = np.cumsum(pdf) / np.sum(pdf)  # Normalize to get cumulative distribution
    threshold = x[np.where(cdf >= alpha)[0][0]]
    return threshold