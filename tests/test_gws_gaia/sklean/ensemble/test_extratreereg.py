from gws_core import BaseTestCase, TaskRunner
from gws_gaia import ExtraTreesRegressorPredictor, ExtraTreesRegressorTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    def test_process(self):
        self.print("Extratrees Regressor")
        table = DataProvider.get_diabetes_table()

        # run trainer
        tester = TaskRunner(
            params={
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
                'nb_estimators': 30
            },
            inputs={'table': table},
            task_type=ExtraTreesRegressorTrainer
        )
        outputs = tester.run()
        trainer_result = outputs['result']

        # run predictior
        test_table = table.select_by_column_names([{"name": "feature.*", "is_regex": True}])
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=ExtraTreesRegressorPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
