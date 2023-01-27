# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import gpboost as gpb
import numpy as np
import pandas as pd
import sklearn
from gws_core import (BoolParam, ConfigParams, FloatRField, InputSpec,
                      ListParam, OutputSpec, ParamSet, StrParam, Table, Task,
                      TaskInputs, TaskOutputs, TechnicalInfo, TextView,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame

from ...base.base_resource import BaseResourceSet
from .helper.lme_design_helper import LMEDesignHelper

# *****************************************************************************
#
# LMEResult
#
# *****************************************************************************

# 1 view summary : gp_model.summary (type text) v
# 1 Ressource predict : gp_model.predict() v
# 1 Ressource Random_effect = gp_model.predict_training_data_random_effects()
# Unique random effects for every group can be obtained as follows
# first_occurences = [np.where(group==i)[0][0] for i in np.unique(group)]
# training_data_random_effects = all_training_data_random_effects.iloc[first_occurences]


@resource_decorator("LMEResult", hide=True)
class LMETrainerResult(BaseResourceSet):
    """ LMEResult """

    PREDICTION_TABLE_NAME = "Prediction table"
    RADOMN_EFFECT_TABLE_NAME = "Random effect table"
    PREDICTION_SCORE_NAME = "Prediction score"

    _prediction_score: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_prediction_table()
            self._create_training_data_random_effects()
            self._create_prediction_score()

    def _create_prediction_table(self) -> DataFrame:
        gp_model = self.get_result()

        training_set = self.get_training_set()
        training_design = self.get_training_design()
        t_mat = LMEDesignHelper.create_training_matrix(training_set=training_set, training_design=training_design)
        n = t_mat.shape[0]
        X_test = np.ones(n)
        Z = LMEDesignHelper.create_design_matrix(training_matrix=t_mat, training_design=training_design)
        df_pred = gp_model.predict(X_pred=X_test, group_data_pred=Z)
        target_pred = pd.DataFrame(df_pred['mu'], columns=['Prediction'])
        df = pd.concat([t_mat, target_pred], axis=1)
        table = Table(df)
        table.name = self.PREDICTION_TABLE_NAME
        self.add_resource(table)

    def _create_training_data_random_effects(self) -> DataFrame:
        gp_model = self.get_result()
        training_set = self.get_training_set()
        df = gp_model.predict_training_data_random_effects()
        training_design = self.get_training_design()
        t_mat = LMEDesignHelper.create_training_matrix(training_set=training_set, training_design=training_design)
        t_mat.index = df.index
        df = gp_model.predict_training_data_random_effects()
        df = pd.concat([t_mat, df], axis=1)
        table = Table(df)
        table.name = self.RADOMN_EFFECT_TABLE_NAME
        self.add_resource(table)

    def _create_prediction_score(self) -> float:
        if not self._prediction_score:
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            t_mat = LMEDesignHelper.create_training_matrix(training_set=training_set, training_design=training_design)
            y_true = t_mat["target"]
            y_pred = t_mat["target"]

            self._prediction_score = sklearn.metrics.r2_score(y_true, y_pred)

        technical_info = TechnicalInfo(key=self.PREDICTION_SCORE_NAME, value=self._prediction_score)
        self.add_technical_info(technical_info)

    def get_prediction_table(self):
        """ Get prediction table """
        if self.resource_exists(self.PREDICTION_TABLE_NAME):
            return self.get_resource(self.PREDICTION_TABLE_NAME)
        else:
            return None

    def get_prediction_score(self):
        return self._prediction_score

    @ view(view_type=TextView, human_name='Summary', short_description='Summary text')
    def view_as_summary(self, params: ConfigParams) -> dict:
        """
        View as summary
        """

        gp_model = self.get_result()
        view_ = TextView(data=str(gp_model.summary()))
        return view_

    # @ view(view_type=TableView, human_name='prediction', short_description='prediction')
    # def view_as_table(self, params: ConfigParams) -> table:
    #     """
    #     View as table
    #     """

    #     table_pred = self.get_result()
    #     view_ = TableView(data=)
    #     return view_


# *****************************************************************************
#
# LMETrainer
#
# *****************************************************************************


@ task_decorator("LMETrainer", human_name="LMETrainer",
                 short_description="Train a linear mixted effects model")
class LMETrainer(Task):
    """
    Trainer of a linear mixted effects model.

    See https://gpboost.readthedocs.io/en/latest for more details
    """
    input_specs = {
        'table': InputSpec(Table, human_name="Table", short_description="Experimental table")
    }
    output_specs = {'result': OutputSpec(LMETrainerResult, human_name="result", short_description="The output result")}
    config_specs = {
        'likelihood': StrParam(default_value="gaussian", allowed_values=["gaussian", "bernoulli_probit", "bernoulli_logit", "poisson", "gamma"]),
        'design': ParamSet({
            'intercept': BoolParam(default_value=True, human_name='Intercept', short_description='Use intercept?'),
            'individual': StrParam(default_value='', human_name='Individual', short_description='The name of the individual observations'),
            'random_effect_structure': ListParam(default_value=[], human_name='Structure of random effects', short_description="The structure of the (nested-)random effects")
        }, human_name="Model design", short_description="The design of the model", max_number_of_occurrences=1),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        training_design = params["design"]
        training_set = inputs["table"]

        # create training matrices
        t_mat = LMEDesignHelper.create_training_matrix(
            training_set=training_set, training_design=training_design)

        # build the matrix of covariables
        Z = LMEDesignHelper.create_design_matrix(
            training_matrix=t_mat, training_design=training_design)

        # create intercept
        n = t_mat.shape[0]
        X = np.ones(n)
        # X_prime=t_mat['time']
        # X_prime=X_prime.to_numpy()
        # X =np.vstack((np.ones(n),X_prime))
        # X=X_prime
        # create GP model
        gp_model = gpb.GPModel(group_data=Z, likelihood=params['likelihood'])
        gp_model.fit(
            y=t_mat["target"],
            X=X,
            params={"std_dev": True}
        )

        result = LMETrainerResult(training_set=training_set, training_design=training_design, result=gp_model)

        return {'result': result}
