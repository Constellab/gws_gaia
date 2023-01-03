from gws_core import (BaseTestCase, ConfigParams, File, Settings, Table,
                      TaskRunner)
from gws_gaia import (DecisionTreeRegressorPredictor,
                      DecisionTreeRegressorTrainer)
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Decision Tree Regressor")
        table = GWSGaiaTestHelper.get_table(index=2, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={
                'max_depth': 4,
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=DecisionTreeRegressorTrainer
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
            task_type=DecisionTreeRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
