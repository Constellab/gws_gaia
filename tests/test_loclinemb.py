
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.loclinemb import Trainer
from gws.settings import Settings
from gws.model import Protocol, Experiment, Study
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
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
        p1.set_param('nb_components', 2)

        def _end(*args, **kwargs):
            r = p1.output['result']

            print(r)

        proto.on_end(_end)
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        
        asyncio.run( e.run() )                  
