from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import LDAPredictor, LDATrainer, LDATransformer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear discriminant analysis classifier")
        dataset = DataProvider.get_iris_dataset()

        # --------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={
                'solver': 'svd',
                'nb_components': 2
            },
            inputs={'dataset': dataset},
            task_type=LDATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # --------------------------------------------------------------
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=LDAPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # --------------------------------------------------------------
        # run transformer
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=LDATransformer
        )
        outputs = await tester.run()
        transformer_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
        print(transformer_result)
