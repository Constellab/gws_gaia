import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import GaussianNaiveBayesTrainer, GaussianNaiveBayesPredictor, GaussianNaiveBayesTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian  Naive Bayes")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset3.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target']
        )

        # run trainer
        tester = TaskTester(
            params = {},
            inputs = {'dataset': dataset},
            task_type = GaussianNaiveBayesTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = GaussianNaiveBayesPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = GaussianNaiveBayesTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)

# class TestTrainer(BaseTestCase):

        
#     async def test_process(self):
#         self.print("Gaussian naive Bayes model")
#         settings = Settings.retrieve()
#         test_dir = settings.get_variable("gws_gaia:testdata_dir")

#         p0 = DatasetLoader()
#         p1 = GaussianNaiveBayesTrainer()
#         p2 = GaussianNaiveBayesPredictor()
#         p3 = GaussianNaiveBayesTester()
        
#         proto = Protocol(
#             processes = {
#                 'p0' : p0,
#                 'p1' : p1,
#                 'p2' : p2,
#                 'p3' : p3                
#             },
#             connectors = [
#         p0>>'dataset' | p1<<'dataset',
#         p0>>'dataset' | p2<<'dataset',
#         p1>>'result' | p2<<'learned_model',
#         p0>>'dataset' | p3<<'dataset',
#         p1>>'result' | p3<<'learned_model'
#             ]
#         )

#         p0.set_param("delimiter", ",")
#         p0.set_param("header", 0)
#         p0.set_param('targets', ['target']) 
#         p0.set_param("file_path", os.path.join(test_dir, "./dataset3.csv"))

#         #p1.set_param('nb_neighbors', 15)

#         experiment: Experiment = Experiment(
#             protocol=proto, study=GTest.study, user=GTest.user)
#         experiment.save()
#         experiment = await ExperimentService.run_experiment(
#             experiment=experiment, user=GTest.user)                    
        
#         r1 = p1.output['result']
#         r2 = p2.output['result']
#         r3 = p3.output['result']
        
#         # print(r1)
#         # print(r2)
#         # print(r3.tuple)