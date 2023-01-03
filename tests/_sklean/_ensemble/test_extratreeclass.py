from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import ExtraTreesClassifierPredictor, ExtraTreesClassifierTrainer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Extratrees Classifier")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
                'nb_estimators': 30
            },
            inputs={'table': table},
            task_type=ExtraTreesClassifierTrainer
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
            task_type=ExtraTreesClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
