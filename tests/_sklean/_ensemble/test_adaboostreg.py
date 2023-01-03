from gws_core import (BaseTestCase, ConfigParams, File, Settings, Table,
                      TableImporter, TaskRunner)
from gws_gaia import AdaBoostRegressorPredictor, AdaBoostRegressorTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("AdaBoost regressor")
        table = GWSGaiaTestHelper.get_table(index=2, header=0, targets=['target'])

        print(table)

        # run trainer
        tester = TaskRunner(
            params={
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
                'nb_estimators': 30
            },
            inputs={'table': table},
            task_type=AdaBoostRegressorTrainer
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
            task_type=AdaBoostRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
