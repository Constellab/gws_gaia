from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import (GradientBoostingClassifierPredictor,
                      GradientBoostingClassifierTrainer)


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gradient Boosting Classifier")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'nb_estimators': 25,
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=GradientBoostingClassifierTrainer
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
            task_type=GradientBoostingClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
