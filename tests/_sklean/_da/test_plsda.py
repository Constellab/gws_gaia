

import numpy
from gws_core import BaseTestCase, TaskRunner
from gws_core.extra import DataProvider as CoreDataProvider
from gws_gaia import PLSDAPredictor, PLSDATrainer


class TestTrainer(BaseTestCase):

    def test_plsda_with_string_targets(self):
        self.print("Partial Least Squares (PLS) regression")
        table = CoreDataProvider.get_iris_table()

        print(table)
        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={
                'nb_components': 3,
                'training_design': [{'target_name': 'variety', 'target_origin': 'column', 'target_type': 'auto'}],
            },
            inputs={'table': table},
            task_type=PLSDATrainer
        )
        outputs = tester.run()
        trainer_result = outputs['result']
        var_table = trainer_result.get_variance_table()
        print(var_table)
        self.assertTrue(numpy.all(numpy.isclose(
            var_table.get_data().to_numpy(), [[0.471143], [0.086125], [0.028790]], atol=1e-3)))

        pred_table = trainer_result.get_prediction_table()
        print(pred_table)
        self.assertEqual(pred_table.shape, (150, 6))

        # ---------------------------------------------------------------------
        test_table = table.select_by_column_names([{"name": "^(?!variety).*", "is_regex": True}])
        print(test_table)
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=PLSDAPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
