
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia import Dataset, DatasetLoader
from gws_gaia import KMeansTrainer, KMeansPredictor
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
        GTest.print("K-means clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = DatasetLoader()
        p2 = KMeansTrainer()
        p3 = KMeansPredictor()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2,
                'p3' : p3
            },
            connectors = [
                p0>>'dataset' | p2<<'dataset',
                p1>>'dataset' | p3<<'dataset',
                p2>>'result' | p3<<'learned_model'
            ]
        )

        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p1.set_param("delimiter", ",")
        p1.set_param("header", 0)
        p1.set_param('targets', ['variety'])
 
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p1.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p2.set_param('nb_clusters', 2)
        
        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)                                 
        
        r1 = p2.output['result']
        r2 = p3.output['result']

        print(r1)
        print(r2)