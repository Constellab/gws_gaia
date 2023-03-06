from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import LDAPredictor, LDATrainer


class TestTrainer(BaseTestCase):

    def test_process(self):
        self.print("Linear discriminant analysis classifier")
        table = DataProvider.get_iris_table(keep_variety=False)

        # --------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={
                'solver': 'svd',
                'nb_components': 2,
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=LDATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # --------------------------------------------------------------
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'table': table,
                'learned_model': trainer_result
            },
            task_type=LDAPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
