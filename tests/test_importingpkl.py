
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.importingpkl import ImporterPKL, Preprocessor, AdhocExtractor
from gws.settings import Settings
from gws.model import Protocol

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        ImporterPKL.drop_table()
        Preprocessor.drop_table()
        AdhocExtractor.drop_table()

    def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        
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

        def _end(*args, **kwargs):
            r1 = p0.output['result']
            r2 = p1.output['result']
            r3 = p2.output['result']
            
            print(r1)
            print(r2)
            print(r3)
            
        proto.on_end(_end)
        e = proto.create_experiment()
        
        asyncio.run( e.run() )
        
        
        
