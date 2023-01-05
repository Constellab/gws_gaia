# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import gpboost as gpb
from gws_core import (BadRequestException, ConfigParams, DataFrameRField,
                      Dataset, FloatParam, FloatRField, InputSpec, IntParam,
                      OutputSpec, Resource, ResourceRField, ScatterPlot2DView,
                      ScatterPlot3DView, StrParam, Table, TableView, Task,
                      TaskInputs, TaskOutputs, TechnicalInfo,
                      resource_decorator, task_decorator, view, CondaShellProxy)
from numpy import ravel
from pandas import DataFrame, concat
from ..base.base_resource import BaseResourceSet

@resource_decorator("LMEResult", hide=True)
class LMEResult(BaseResourceSet):
    """ LMEResult """

# *****************************************************************************
#
# LMEPREPROCESSING
#
# *****************************************************************************


@task_decorator("LMETrainer", human_name="LMETrainer",
                short_description="Train a linear mixted effects model")