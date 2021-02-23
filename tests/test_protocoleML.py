
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.linearda import Trainer as LDATrainer
from gaia.linearda import Predictor as LDAPredictor
from gaia.linearda import Tester as LDATester
from gaia.linearda import Transformer as LDATransformer
from gaia.pca import Trainer as PCATrainer
from gaia.pca import Transformer as PCATransformer

from gws.settings import Settings
from gws.model import Protocol

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        LDATrainer.drop_table()
        LDAPredictor.drop_table()
        LDATester.drop_table()

    def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")

        p0 = Importer()
        p1 = LDATrainer()
        p2 = LDAPredictor()
        p3 = LDATester()
        p4 = LDATransformer()
        p5 = PCATrainer()
        p6 = PCATransformer()
        
        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2,
                'p3' : p3,
                'p4' : p4,
                'p5' : p5,
                'p6' : p6
            },
            connectors = [
        p0>>'dataset' | p5<<'dataset',                
        p0>>'dataset' | p6<<'dataset',                
        p5>>'result' | p6<<'learned_model',                
        p0>>'dataset' | p1<<'dataset',
        p0>>'dataset' | p4<<'dataset',
        p1>>'result' | p4<<'learned_model',
        p0>>'dataset' | p3<<'dataset',
        p1>>'result' | p3<<'learned_model',
        p0>>'dataset' | p2<<'dataset',
        p1>>'result' | p2<<'learned_model',
            ]
        )

        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety']) 
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))

        p1.set_param('nb_components', 2)

        p5.set_param('nb_components', 2)

        def _end(*args, **kwargs):
            r1 = p1.output['result']
            r2 = p2.output['result']
            r3 = p3.output['result']
            r4 = p4.output['result']
            r5 = p5.output['result']
            r6 = p6.output['result']
            # print(r1)
            # print(r2)
            # print(r4.tuple)
            # print(r6.tuple)   
            
        proto.on_end(_end)
        e = proto.create_experiment()
        
        asyncio.run( e.run() )                 