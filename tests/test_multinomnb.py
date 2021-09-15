import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import MultinomialNaiveBayesClassifierTrainer, MultinomialNaiveBayesClassifierPredictor, MultinomialNaiveBayesClassifierTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        GTest.print("Naive Bayes classifier for a multinomial model")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = MultinomialNaiveBayesClassifierTrainer()
        p2 = MultinomialNaiveBayesClassifierPredictor()
        p3 = MultinomialNaiveBayesClassifierTester()
        
        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2,
                'p3' : p3                
            },
            connectors = [
        p0>>'dataset' | p1<<'dataset',
        p0>>'dataset' | p2<<'dataset',
        p1>>'result' | p2<<'learned_model',
        p0>>'dataset' | p3<<'dataset',
        p1>>'result' | p3<<'learned_model'
            ]
        )

        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['target']) 
        p0.set_param("file_path", os.path.join(test_dir, "./dataset7.csv"))
        p1.set_param('alpha', 1) 
        
        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)               
        
        r1 = p1.output['result']
        r2 = p2.output['result']
        r3 = p3.output['result']
        
        # print(r1)
        # print(r2)
        print(r3.tuple)
