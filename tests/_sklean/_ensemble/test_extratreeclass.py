from gws_core import BaseTestCase, TaskRunner
from gws_core.extra import DataProvider
from gws_gaia import ExtraTreesClassifierPredictor, ExtraTreesClassifierTrainer


class TestTrainer(BaseTestCase):

    def test_process(self):
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
        outputs = tester.run()
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
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
