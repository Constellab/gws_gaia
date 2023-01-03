from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import LogisticRegressionPredictor, LogisticRegressionTrainer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Logistic regression classifier")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'inv_reg_strength': 1e5,
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=LogisticRegressionTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        test_table = table.select_by_column_names([{"name": "^(?!variety).*", "is_regex": True}])
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=LogisticRegressionPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
