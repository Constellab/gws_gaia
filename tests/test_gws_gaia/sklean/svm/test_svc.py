from gws_core import BaseTestCase, TaskRunner
from gws_core.extra import DataProvider
from gws_gaia import SVCPredictor, SVCTrainer


class TestTrainer(BaseTestCase):

    def test_process(self):
        self.print("C-Support Vector Classifier (SVC)")
        table = DataProvider.get_iris_table(keep_variety=False)

        # run trainer
        tester = TaskRunner(
            params={
                'probability': True,
                'training_design': [{'target_name': 'variety', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=SVCTrainer
        )
        outputs = tester.run()
        trainer_result = outputs['result']

        # run predictior
        test_table = table.select_by_column_names([{"name": "^(?!variety).*", "is_regex": True}])
        tester = TaskRunner(
            params={},
            inputs={
                'table': test_table,
                'learned_model': trainer_result
            },
            task_type=SVCPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
