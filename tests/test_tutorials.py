
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_core import Settings, GTest, Protocol, Experiment, ExperimentService
from gws_gaia.tutorials.lda_pca import lda_pca_experiment

class TestTrainer(IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()

    async def test_process(self):
        GTest.print("Small tutorial")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        data_file = os.path.join(test_dir, "./iris.csv")
        e = lda_pca_experiment(data_file, delimiter=",", header=0, target=['variety'], ncomp=2)
        experiment = await ExperimentService.run_experiment(
            experiment=e, user=GTest.user)
                     