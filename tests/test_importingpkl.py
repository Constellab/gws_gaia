import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia import Dataset, DatasetLoader
from gws_gaia.tf import ImporterPKL, Preprocessor, AdhocExtractor
from gws_core import Settings, GTest, Protocol, Experiment, ExperimentService

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
        GTest.print("ImporterPKL")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        
        p0 = ImporterPKL()
        p1 = Preprocessor()
        p2 = AdhocExtractor()
        
        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2
            },
            connectors = [
                p0>>'result' | p1<<'data',
                p0>>'result' | p2<<'data',
            ]
        )
        
        p0.set_param("file_path", os.path.join(test_dir, "./mnist.pkl"))
        p1.set_param('number_classes', 10)

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)
        
        r1 = p0.output['result']
        r2 = p1.output['result']
        r3 = p2.output['result']
        
        print(r1)
        print(r2)
        print(r3)
        
