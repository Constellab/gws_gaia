
import os
import asyncio

from gws_gaia import Dataset, DatasetLoader
from gws_core import (Settings, GTest, TaskTester, 
                        ConfigParams, TaskInputs, BaseTestCase)

class TestImporter(BaseTestCase):
    
    async def test_importer(self):
        GTest.print("Dataset import")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        # run trainer
        tester = TaskTester(
            params = ConfigParams({
                "file_path": os.path.join(test_dir, "./iris.csv"),
                "delimiter": ",",
                "header": 0,
                "targets": ["variety"]
            }),
            inputs = TaskInputs(),
            task = DatasetLoader()
        )
        outputs = await tester.run()
        ds = outputs['dataset']
        self.assertEquals(ds.nb_features, 4)
        self.assertEquals(ds.nb_targets, 1)
        self.assertEquals(ds.nb_instances, 150)
        self.assertEquals(ds.features.values[0,0], 5.1)
        self.assertEquals(ds.features.values[0,1], 3.5)
        self.assertEquals(ds.features.values[149,0], 5.9)
        self.assertEquals(list(ds.feature_names), ["sepal.length","sepal.width","petal.length","petal.width"])
        self.assertEquals(list(ds.target_names), ["variety"])
        

    async def test_importer_no_head(self):
        GTest.print("Dataset import without header")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        # run trainer
        tester = TaskTester(
            params = ConfigParams({
                "file_path": os.path.join(test_dir, "./iris_no_head.csv"),
                "delimiter": ",",
                "header": None,
                "targets": [4]
            }),
            inputs = TaskInputs(),
            task = DatasetLoader()
        )
        outputs = await tester.run()
        ds = outputs['dataset']
        self.assertEquals(ds.nb_features, 4)
        self.assertEquals(ds.nb_targets, 1)
        self.assertEquals(ds.nb_instances, 150)
        self.assertEquals(ds.features.values[0,0], 5.1)
        self.assertEquals(ds.features.values[0,1], 3.5)
        self.assertEquals(ds.features.values[149,0], 5.9)
        self.assertEquals(list(ds.feature_names), list(range(0,4)))
        self.assertEquals(list(ds.target_names), [4])

        print(ds)