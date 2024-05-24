from gws_core import BaseTestCase, TaskRunner
from gws_core.extra import DataProvider
from gws_gaia import AdaBoostClassifierPredictor, AdaBoostClassifierTrainer


class TestTrainer(BaseTestCase):

    def test_adaboost_process(self):
        self.print("AdaBoost classifier")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
                'nb_estimators': 30
            },
            inputs={'table': table},
            task_type=AdaBoostClassifierTrainer
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
            task_type=AdaBoostClassifierPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
