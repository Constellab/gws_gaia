
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.ridgeclass import Trainer, Predictor, Tester
from gws.settings import Settings
from gws.model import Protocol, Experiment, Job, Study

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Predictor.drop_table()
        Tester.drop_table()
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()
        Predictor.drop_table()
        Tester.drop_table()
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")

        p0 = Importer()
        p1 = Trainer()
        p2 = Predictor()
        p3 = Tester()
        
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
        p0.set_param("file_path", os.path.join(test_dir, "./diabetes.csv"))

        p1.set_param('alpha', 1) 

        def _end(*args, **kwargs):
            r1 = p1.output['result']
            r2 = p2.output['result']
            r3 = p3.output['result']
            
            # print(r1)
            # print(r2)
            print(r3.tuple)

        proto.on_end(_end)
        e = proto.create_experiment(study=Study.get_default_instance())
        
        asyncio.run( e.run() )               
        
        
