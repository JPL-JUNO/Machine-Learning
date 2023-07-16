"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-16 20:49:50
"""

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame


def tpr_fpr_dataframe(y_true: ndarray, y_pred: ndarray) -> DataFrame:
    scores = []
    thresholds = np.linspace(0, 1, 101)
    for threshold in thresholds:
        tp = ((y_pred >= threshold) & (y_true == 1)).sum()
        fp = ((y_pred >= threshold) & (y_true == 0)).sum()
        fn = ((y_pred < threshold) & (y_true == 1)).sum()
        tn = ((y_pred < threshold) & (y_true == 0)).sum()
        scores.append((threshold, tp, fp, fn, tn))
    df_scores = pd.DataFrame(
        scores, columns=['Threshold', 'TP', 'FP', 'FN', 'TN'])
    df_scores['TPR'] = df_scores['TP'] / (df_scores['TP'] + df_scores['FN'])
    df_scores['FPR'] = df_scores['FP'] / (df_scores['FP'] + df_scores['TN'])
    return df_scores
