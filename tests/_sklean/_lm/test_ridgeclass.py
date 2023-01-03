from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import RidgeClassifierPredictor, RidgeClassifierTrainer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Ridge classifier model")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'alpha': 1,
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=RidgeClassifierTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'table': table,
                'learned_model': trainer_result
            },
            task_type=RidgeClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
