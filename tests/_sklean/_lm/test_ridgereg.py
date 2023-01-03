from gws_core import (BaseTestCase, ConfigParams, Table, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import RidgeRegressionPredictor, RidgeRegressionTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Ridge regression model")
        table = DataProvider.get_diabetes_table()

        # run trainer
        tester = TaskRunner(
            params={
                'alpha': 1,
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=RidgeRegressionTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        test_table = table.select_by_column_names([{"name": "feature.*", "is_regex": True}])
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=RidgeRegressionPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
