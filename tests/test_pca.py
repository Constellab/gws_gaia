
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.pca import Trainer
from gws.settings import Settings
from gws.model import Protocol

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
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        
        p0 = Importer(instance_name="p0")
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        
        p1 = Trainer(instance_name="p1")
        p1.set_param('nb_components', 2)
        
        proto = Protocol(
            processes = {
                "p0": p0,
                "p1": p1
            },
            connectors=[
                p0>>'dataset' | p1<<'dataset'
            ]
        )
        
        e = proto.create_experiment()
        await e.run()

        r = p1.output['result']
