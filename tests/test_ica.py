import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.ica import Trainer
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
        GTest.print("Independant Component Analysis (ICA)")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")

        p0 = Importer()
        p1 = Trainer()

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
        p1.set_param('nb_components', 7)

        def _end(*args, **kwargs):
            r = p1.output['result']

            print(r)

        
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_end)
        asyncio.run( e.run() )                                         
