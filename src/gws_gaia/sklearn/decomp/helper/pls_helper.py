


from typing import List

import numpy as np
from gws_core import Table
from pandas import DataFrame
from sklearn.metrics import r2_score

from ....base.helper.training_design_helper import TrainingDesignHelper


class PLSHelper():

    @classmethod
    def create_variance_table(cls, pls, training_set, training_design, dummy=False) -> List[float]:
        _, y_true = TrainingDesignHelper.create_training_matrices(training_set, training_design, dummy=dummy)
        y_std = y_true.std(axis=0, ddof=1)
        y_mean = y_true.mean(axis=0)

        r2_list = []
        ncomp = pls.x_scores_.shape[1]
        for i in range(0, ncomp):
            y_pred = np.dot(
                pls.x_scores_[:, i].reshape(-1, 1),
                pls.y_loadings_[:, i].reshape(-1, 1).T)
            y_pred = DataFrame(y_pred)
            for k in range(0, y_true.shape[1]):
                y_pred.iloc[:, k] = y_pred.iloc[:, k] * y_std.iat[k] + y_mean.iat[k]

            r2_list.append(r2_score(y_true, y_pred))

        index = [f"PC{n+1}" for n in range(0, ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(r2_list, columns=columns, index=index)
        table = Table(data=data)

        return table
