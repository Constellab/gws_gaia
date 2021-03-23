
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer
from gaia.pca import Trainer, Transformer
from gws.settings import Settings
from gws.model import Protocol, Experiment, Job, Study

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
        p2 = Transformer()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2
            },
            connectors = [
                p0>>'dataset' | p1<<'dataset',
                p0>>'dataset' | p2<<'dataset',
                p1>>'result' | p2<<'learned_model',
            ]
        )
        
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p1.set_param('nb_components', 2)

        def _end(*args, **kwargs):
            r1 = p1.output['result']
            r2 = p2.output['result']

            #print(r2.tuple)

        proto.on_end(_end)
        e = proto.create_experiment(study=Study.get_default_instance())
        
        asyncio.run( e.run() )               