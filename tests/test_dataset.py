
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia import Dataset, DatasetLoader
from gws_core import Settings, GTest, Protocol, Experiment, ExperimentService

class TestImporter(IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    async def test_importer(self):
        GTest.print("Dataset")
        p0 = DatasetLoader(instance_name="p0")
        settings = Settings.retrieve()

        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param("targets", ["variety"])
            
        experiment: Experiment = ExperimentService.create_experiment_from_process(process=p0)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)
        
        ds = p0.output['dataset']
        self.assertEquals(ds.nb_features, 4)
        self.assertEquals(ds.nb_targets, 1)
        self.assertEquals(ds.nb_instances, 150)
        self.assertEquals(ds.features.values[0,0], 5.1)
        self.assertEquals(ds.features.values[0,1], 3.5)
        self.assertEquals(ds.features.values[149,0], 5.9)
        self.assertEquals(list(ds.feature_names), ["sepal.length","sepal.width","petal.length","petal.width"])
        self.assertEquals(list(ds.target_names), ["variety"])
        

    async def test_importer_no_head(self):
        p0 = DatasetLoader(instance_name="p0")
        settings = Settings.retrieve()

        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris_no_head.csv"))
        p0.set_param("header", None)
        p0.set_param("delimiter", ",")
        p0.set_param("targets", [4])

        experiment: Experiment = ExperimentService.create_experiment_from_process(process=p0)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)
        
        ds = p0.output['dataset']
        self.assertEquals(ds.nb_features, 4)
        self.assertEquals(ds.nb_targets, 1)
        self.assertEquals(ds.nb_instances, 150)
        self.assertEquals(ds.features.values[0,0], 5.1)
        self.assertEquals(ds.features.values[0,1], 3.5)
        self.assertEquals(ds.features.values[149,0], 5.9)
        self.assertEquals(list(ds.feature_names), list(range(0,4)))
        self.assertEquals(list(ds.target_names), [4])

        print(ds)