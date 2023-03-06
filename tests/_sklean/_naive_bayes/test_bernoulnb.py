from gws_core import BaseTestCase, TaskRunner
from gws_gaia import (BernoulliNaiveBayesClassifierPredictor,
                      BernoulliNaiveBayesClassifierTrainer)
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    def test_process(self):
        self.print("Naive Bayes classifier")
        table = GWSGaiaTestHelper.get_table(index=7, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={
                'alpha': 1,
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=BernoulliNaiveBayesClassifierTrainer
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
            task_type=BernoulliNaiveBayesClassifierPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
