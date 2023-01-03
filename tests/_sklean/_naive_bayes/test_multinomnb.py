from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_gaia import (MultinomialNaiveBayesClassifierPredictor,
                      MultinomialNaiveBayesClassifierTrainer)
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Naive Bayes classifier for a multinomial model")
        table = GWSGaiaTestHelper.get_table(index=7, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={
                'alpha': 1,
                'training_design': [{'target_name': 'target', 'target_origin': 'column'}],
            },
            inputs={'table': table},
            task_type=MultinomialNaiveBayesClassifierTrainer
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
            task_type=MultinomialNaiveBayesClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
