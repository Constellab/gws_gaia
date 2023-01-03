from gws_core import (BaseTestCase, ConfigParams, Table, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import (RandomForestRegressorPredictor,
                      RandomForestRegressorTrainer)
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Random forest regressor")
        table = DataProvider.get_diabetes_table()

        # run trainer
        tester = TaskRunner(
            params={
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=RandomForestRegressorTrainer
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
            task_type=RandomForestRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
