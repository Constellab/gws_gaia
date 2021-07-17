
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.kmeans import Trainer, Predictor
from gws.settings import Settings
from gws.protocol import Protocol
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    def test_process(self):
        GTest.print("K-means clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")

        p0 = Importer()
        p1 = Importer()
        p2 = Trainer()
        p3 = Predictor()

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
        
        def _end(*args, **kwargs):
            r1 = p2.output['result']
            r2 = p3.output['result']

            print(r1)
            print(r2)

        
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_end)
        asyncio.run( e.run() )                                 
        
