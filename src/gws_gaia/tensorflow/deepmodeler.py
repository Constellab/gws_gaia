# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)

from tensorflow.keras import Model as KerasModel

from .data import DeepModel, DeepResult, Tensor

# *****************************************************************************
#
# DeepModelBuilder
#
# *****************************************************************************


@task_decorator("DeepModelBuilder", human_name="Deep modeler builder",
                short_description="Groups layers into a model")
class DeepModelBuilder(Task):
    """
    Build the model from layers specifications
    """
    input_specs = {'inputs': InputSpec(Tensor, human_name="Tensor", short_description="The input tensors"),
        'outputs': InputSpec(Tensor, human_name="Tensor", short_description="The input tensors")}
    output_specs = {'result': OutputSpec(DeepModel, human_name="Result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['inputs']
        y = inputs['outputs']
        x1 = x.get_result()
        y1 = y.get_result()
        z = KerasModel(inputs=x1, outputs=y1)
        result = DeepModel(model=z)
        return {'result': result}

# *****************************************************************************
#
# DeepModelCompiler
#
# *****************************************************************************


@task_decorator("DeepModelCompiler", human_name="Deep model compiler",
                short_description="Configures a model for training")
class DeepModelCompiler(Task):
    """
    Configures the model for training.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'builded_model': InputSpec(DeepModel, human_name="Model", short_description="The input model")}
    output_specs = {'result': OutputSpec(DeepModel, human_name="Result", short_description="The output model")}
    config_specs = {
        'optimizer': StrParam(default_value='rmsprop'),
        'loss': StrParam(default_value=''),
        'metrics': StrParam(default_value='')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['builded_model']
        model = x.get_result()
        model.compile(optimizer=params["optimizer"], loss=params["loss"], metrics=params["metrics"])
        result = DeepModel(model=model)
        return {'result': result}

# *****************************************************************************
#
# DeepModelerTrainer
#
# *****************************************************************************


@task_decorator("DeepModelerTrainer")
class DeepModelerTrainer(Task):
    """
    Trainer of a model on a dataset

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset': InputSpec(DeepResult, human_name="Dataset", short_description="The input dataset"),
        'compiled_model': InputSpec(DeepModel, human_name="Model", short_description="The input model")}
    output_specs = {'result': OutputSpec(DeepModel, human_name="Result", short_description="The output result")}
    config_specs = {
        'batch_size': IntParam(default_value=32, min_value=0),
        'epochs': IntParam(default_value=1, min_value=0),
        'validation_split': FloatParam(default_value=0.1, min_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['compiled_model']
        model = x.get_result()
        y = inputs['dataset']
        data = y.get_result()
        (x_train, y_train), (_, _) = data
        model.fit(
            x_train, y_train, batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_split=params["validation_split"])
        result = DeepModel(model=model)
        return {'result': result}

# *****************************************************************************
#
# DeepModelerPredictor
#
# *****************************************************************************


@task_decorator("DeepModelerPredictor")
class DeepModelerPredictor(Task):
    """
    Predictor of a trained model from a dataset. Generates output predictions for the input samples.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset': InputSpec(DeepResult, human_name="Dataset", short_description="The input dataset"),
        'trained_model': InputSpec(DeepModel, human_name="Model", short_description="The input model")}
    output_specs = {'result': OutputSpec(DeepResult, human_name="Result", short_description="The output result")}
    config_specs = {
        'verbosity_mode': IntParam(default_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['trained_model']
        model = x.get_result()
        y = inputs['dataset']
        data = y.get_result()
        result = model.predict(data, verbose=params['verbosity_mode'])
        result = DeepResult(result=result)
        return {'result': result}
