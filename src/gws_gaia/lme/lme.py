# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import gpboost as gpb
import numpy as np
import pandas as pd
from gws_core import (BadRequestException, BoolParam, CondaShellProxy,
                      ConfigParams, DataFrameRField, Dataset, FloatParam,
                      FloatRField, InputSpec, IntParam, ListParam, OutputSpec,
                      ParamSet, Resource, ResourceRField, ScatterPlot2DView,
                      ScatterPlot3DView, StrParam, Table,
                      TableTagExtractorHelper, TableView, Task, TaskInputs,
                      TaskOutputs, TechnicalInfo, TextView, resource_decorator,
                      task_decorator, view)

from ..base.base_resource import BaseResourceSet
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

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)

    @view(view_type=TextView, human_name='Summary', short_description='Summary text')
    def view_as_summary(self, params: ConfigParams) -> dict:
        """
        View as summary
        """

        gp_model = self.get_result()
        view_ = TextView(data=str(gp_model.summary()))
        return view_

    # @view(view_type=TableView, human_name="Prediction table")
    # def view_predictions_as_table(self, params: ConfigParams) -> dict:
    #     """
    #     View the target data and the predicted data in a table. Works for data with only one target
    #     """
    #     Y_data = self._training_set.get_targets()
    #     Y_predicted = self._get_predicted_data()
    #     y = pd.concat([Y_data, Y_predicted], axis=1)
    #     data = y.set_axis(["YData", "YPredicted"], axis=1)
    #     t_view = TableView(Table(data))
    #     return t_view

# *****************************************************************************
#
# LMETrainer
#
# *****************************************************************************


@task_decorator("LMETrainer", human_name="LMETrainer",
                short_description="Train a linear mixted effects model")
class LMETrainer(Task):
    """
    Trainer of a linear mixted effects model.

    See https://gpboost.readthedocs.io/en/latest for more details
    """
    input_specs = {
        'dataset': InputSpec(Dataset, human_name="Dataset", short_description="Experimental dataset")
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
        training_set = inputs["dataset"]

        # create training matrices
        t_mat = LMEDesignHelper.create_training_matrix(
            training_set=training_set, training_design=training_design)

        # build the matrix of covariables
        Z = LMEDesignHelper.create_design_matrix(
            training_matrix=t_mat, training_design=training_design)

        # create intercept
        n = t_mat.shape[0]
        X = np.ones(n)

        # create GP model
        gp_model = gpb.GPModel(group_data=Z, likelihood=params['likelihood'])
        gp_model.fit(
            y=t_mat["target"],
            X=X,
            params={"std_dev": True}
        )

        result = LMETrainerResult(training_set=Dataset, result=gp_model)

        return {'result': result}

# *****************************************************************************
#
# LinearMixedEffectsPredictor
#
# *****************************************************************************


# @task_decorator("LMEPredictor", human_name="Linear regression predictor",
#                 short_description="Predict dataset targets using a trained lme model")

# class LMEPredictor(Task):
#     input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
#                    'learned_model': InputSpec(LMETrainerResult, human_name="Learned model", short_description="The input model")}
#     output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
#     config_specs = {}

#     async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
#         dataset = inputs['dataset']
#         learned_model = inputs['learned_model']
#         lmr = learned_model.get_result()
#         y = lmr.predict(dataset.get_features().values)
#         print(y)
#         result_dataset = Dataset(
#             data=pd.DataFrame(y),
#             row_names=dataset.row_names,
#             column_names=dataset.target_names,
#             target_names=dataset.target_names
#         )
#         return {'result': result_dataset}
