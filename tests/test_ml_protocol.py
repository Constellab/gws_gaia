
import os
import asyncio
import unittest

from gws.model import Process, Resource

from gws.settings import Settings
from gaia._tuto.tutorial import lda_pca_experiment

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Process.drop_table()
        Resource.drop_table()

    def test_process(self):
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        
        data_file = os.path.join(test_dir, "./iris.csv")
        e = lda_pca_experiment(data_file, delimiter=",", header=0, target=['variety'], ncomp=2)
        asyncio.run( e.run() )
                     