
import os
import asyncio
import unittest

from gaia.data.dataset import Dataset, Importer as DatasetImporter
from gws.settings import Settings

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Dataset.drop_table()

    def test_importer(self):
        
        async def _import_iris(self):
            p0 = DatasetImporter()
            settings = Settings.retrieve()

            test_dir = settings.get_dir("gaia:testdata_dir")
            p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
            p0.set_param("delimiter", ",")
            p0.set_param("header", 0)
            p0.set_param("targets", ["variety"])
            await p0.run()

            ds = p0.output['dataset']
            self.assertEquals(ds.nb_features, 4)
            self.assertEquals(ds.nb_targets, 1)
            self.assertEquals(ds.nb_instances, 150)
            self.assertEquals(ds.features.values[0,0], 5.1)
            self.assertEquals(ds.features.values[0,1], 3.5)
            self.assertEquals(ds.features.values[149,0], 5.9)
            
            self.assertEquals(list(ds.feature_names), ["sepal.length","sepal.width","petal.length","petal.width"])
            self.assertEquals(list(ds.target_names), ["variety"])

        asyncio.run( _import_iris(self) )
        
    def test_importer_no_head(self):

        async def _import_iris_no_head(self):
            p0 = DatasetImporter()
            settings = Settings.retrieve()

            test_dir = settings.get_dir("gaia:testdata_dir")
            p0.set_param("file_path", os.path.join(test_dir, "./iris_no_head.csv"))
            p0.set_param("delimiter", ",")
            p0.set_param("targets", [4])
            await p0.run()

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
            
        asyncio.run( _import_iris_no_head(self) )

        