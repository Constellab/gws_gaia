
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner, ViewTester)
from gws_gaia import PLSPredictor, PLSTrainer, PLSTransformer
from gws_gaia.extra import DataProvider
from gws_core.extra import DataProvider as CoreDataProvider

class TestTrainer(BaseTestCase):

    async def test_pls_with_numeric_targets(self):
        self.print("Partial Least Squares (PLS) regression")
        dataset = DataProvider.get_diabetes_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 3},
            inputs={'dataset': dataset},
            task_type=PLSTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        table = trainer_result.get_variance_table()
        print(table)
        self.assertTrue(numpy.all(numpy.isclose(
            table.get_data().to_numpy(), [[0.406361], [0.101965], [0.005088]], atol=1e-3)))

        table = trainer_result.get_prediction_table()
        print(table)

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


    async def test_pls_with_straing_targets(self):
        self.print("Partial Least Squares (PLS) regression")
        dataset = CoreDataProvider.get_iris_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 3},
            inputs={'dataset': dataset},
            task_type=PLSTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        table = trainer_result.get_variance_table()
        print(table)
        # self.assertTrue(numpy.all(numpy.isclose(
        #     table.get_data().to_numpy(), [[0.406361], [0.101965], [0.005088]], atol=1e-3)))

        table = trainer_result.get_prediction_table()
        print(table)

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
