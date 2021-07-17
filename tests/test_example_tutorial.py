
import os
import asyncio
import unittest

from gws.protocol import Protocol
from gws.settings import Settings
from gaia.example.tutorial import lda_pca_experiment
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()

    def test_process(self):
        GTest.print("Small tutorial")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        data_file = os.path.join(test_dir, "./iris.csv")
        e = lda_pca_experiment(data_file, delimiter=",", header=0, target=['variety'], ncomp=2)
        asyncio.run( e.run() )
                     