
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner, ViewTester)
from gws_gaia import PLSPredictor, PLSTrainer, PLSTransformer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Partial Least Squares (PLS) regression")
        dataset = DataProvider.get_diabetes_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'dataset': dataset},
            task_type=PLSTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # ---------------------------------------------------------------------
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=PLSPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # ---------------------------------------------------------------------
        # run tester
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=PLSTransformer
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
        print(tester_result)
