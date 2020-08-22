
import os
import asyncio
import unittest

from gaia.data.dataset import Dataset, Importer
from gaia.comp.pca import Trainer
from gws.settings import Settings

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()
        Trainer.drop_table()

    def test_process(self):
        asyncio.run( self._process() )

    async def _process(self):
        p0 = Importer()
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p1 = Trainer()
        
        p0>>'dataset' | p1<<'dataset'
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p1.set_param('nb_components', 2)
        await p0.run()

        r = p1.output['result']
