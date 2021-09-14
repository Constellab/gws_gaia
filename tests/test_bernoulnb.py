
import os
import asyncio

from gws_gaia import Dataset
from gws_gaia import (BernoulliNaiveBayesClassifierTrainer, BernoulliNaiveBayesClassifierPredictor, 
                        BernoulliNaiveBayesClassifierTester)
from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        GTest.print("Naive Bayes classifier")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset7.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target']
        )

        # run trainer
        tester = TaskTester(
            params = ConfigParams({'alpha': 1}),
            inputs = TaskInputs({'dataset': dataset}),
            task = BernoulliNaiveBayesClassifierTrainer()
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = BernoulliNaiveBayesClassifierPredictor()
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = BernoulliNaiveBayesClassifierTester()
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
