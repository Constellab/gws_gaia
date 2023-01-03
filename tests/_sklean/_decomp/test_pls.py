
import os

import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_core.extra import DataProvider as CoreDataProvider
from gws_gaia import PLSPredictor, PLSTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_plsr_with_numeric_targets(self):
        self.print("Partial Least Squares (PLS) regression")
        table = DataProvider.get_diabetes_table()

        print(table)

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={
                'nb_components': 3,
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=PLSTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        var_table = trainer_result.get_variance_table()
        print(var_table)
        self.assertTrue(numpy.all(numpy.isclose(
            var_table.get_data().to_numpy(), [[0.406361], [0.101965], [0.005088]], atol=1e-3)))

        var_table = trainer_result.get_prediction_table()
        print(var_table)

        # ---------------------------------------------------------------------
        test_table = table.select_by_column_names([{"name": "feature.+", "is_regex": True}])
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=PLSPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
