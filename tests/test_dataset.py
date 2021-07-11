
import os
import asyncio
import unittest

from gaia.dataset import Dataset, Importer as DatasetImporter
from gws.settings import Settings
from gws.protocol import Protocol
from gws.unittest import GTest

class TestImporter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    def test_importer(self):
        
        p0 = DatasetImporter(instance_name="p0")
        settings = Settings.retrieve()

        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param("targets", ["variety"])
            
        def _on_end(*args, **kwargs):  
            ds = p0.output['dataset']
            self.assertEquals(ds.nb_features, 4)
            self.assertEquals(ds.nb_targets, 1)
            self.assertEquals(ds.nb_instances, 150)
            self.assertEquals(ds.features.values[0,0], 5.1)
            self.assertEquals(ds.features.values[0,1], 3.5)
            self.assertEquals(ds.features.values[149,0], 5.9)

            self.assertEquals(list(ds.feature_names), ["sepal.length","sepal.width","petal.length","petal.width"])
            self.assertEquals(list(ds.target_names), ["variety"])
        
        e = p0.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_on_end)
        asyncio.run( e.run() )
        
    def test_importer_no_head(self):

        p0 = DatasetImporter(instance_name="p0")
        settings = Settings.retrieve()

        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris_no_head.csv"))
        p0.set_param("delimiter", ",")
        p0.set_param("targets", [4])
            
        def _on_end(*args, **kwargs):  
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
            
        e = p0.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_on_end)
        asyncio.run( e.run() )
        