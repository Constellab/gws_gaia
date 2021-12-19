from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, IExperiment, IProtocol, ProcessSpec, Protocol,
                      Settings, TaskInputs, TaskRunner, protocol_decorator)
from gws_core.extra import DataProvider
from gws_gaia import AdaBoostClassifierPredictor, AdaBoostClassifierTrainer


class TestTrainer(BaseTestCase):

    async def test_adaboost_process(self):
        self.print("AdaBoost classifier")
        dataset = DataProvider.get_iris_dataset()

        # run trainer
        tester = TaskRunner(
            params={'nb_estimators': 30},
            inputs={'dataset': dataset},
            task_type=AdaBoostClassifierTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=AdaBoostClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
