
import os
import asyncio
import unittest

from gws.model import Protocol, Experiment, Job, Study

from gws.settings import Settings
from gaia._tuto.tutorial import lda_pca_experiment

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()

    @classmethod
    def tearDownClass(cls):
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()

    def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        
        data_file = os.path.join(test_dir, "./iris.csv")
        e = lda_pca_experiment(data_file, delimiter=",", header=0, target=['variety'], ncomp=2)
        asyncio.run( e.run() )
                     