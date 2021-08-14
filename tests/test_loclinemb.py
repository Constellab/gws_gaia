
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia import Dataset, DatasetLoader
from gws_gaia import LocallyLinearEmbeddingTrainer
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
        GTest.print("Locally linear embedding model")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = LocallyLinearEmbeddingTrainer()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1
            },
            connectors = [
                p0>>'dataset' | p1<<'dataset',
            ]
        )
        
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', [])
        p0.set_param("file_path", os.path.join(test_dir, "./digits.csv"))
        p1.set_param('nb_components', 2)

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)                  

        r = p1.output['result']
        print(r)