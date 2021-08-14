import os
import asyncio
import time
from unittest import IsolatedAsyncioTestCase

from gws_core import Settings, GTest, Protocol, Experiment, ExperimentService
from gws_gaia import Dataset, DatasetLoader
from gws_gaia import AdaBoostClassifierTrainer, AdaBoostClassifierPredictor, AdaBoostClassifierTester

class TestTrainer(IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()

    async def test_process(self):
        GTest.print("AdaBoost classifier")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        
        p0 = DatasetLoader()
        p1 = AdaBoostClassifierTrainer()
        p2 = AdaBoostClassifierPredictor()
        p3 = AdaBoostClassifierTester()

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
                p1>>'result' | p3<<'learned_model',
                p0>>'dataset' | p3<<'dataset'
            ]
        )
        
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p1.set_param('nb_estimators', 30)

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()

        try:
            experiment = await ExperimentService.run_experiment(
                experiment=experiment, user=GTest.user)
        except Exception as err:
            print(err)

        self.assertTrue(experiment.is_finished)
        self.assertTrue(p1.is_finished)
        r1 = p1.output['result']
        r2 = p2.output['result']
        r3 = p3.output['result'] 

        print(r1)
        print(r2)
        print(r3.tuple)


        
        
        
        
