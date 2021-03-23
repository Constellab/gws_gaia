
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.aggclust import Trainer
from gws.settings import Settings
from gws.model import Protocol, Study, Experiment, Job

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Protocol.drop_table()
        Job.drop_table()
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
        p0.set_param('targets', ['target1','target2'])
        p0.set_param("file_path", os.path.join(test_dir, "./dataset1.csv"))
        p1.set_param('nb_clusters', 2)

        def _end(*args, **kwargs):
            r1 = p1.output['result']
            
            print(r1)

        proto.on_end(_end)
        e = proto.create_experiment(study=Study.get_default_instance())
        
        asyncio.run( e.run() )                
