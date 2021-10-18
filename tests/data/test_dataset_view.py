import os
from gws_core import (Settings, TaskTester, BaseTestCase)
from gws_gaia import Dataset, DatasetView, DatasetLoader


class TestTDatasetView(BaseTestCase):

    async def test_dataset_view(self,):
        self.print("Dataset import")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        # run trainer
        tester = TaskTester(
            params = {
                "file_path": os.path.join(test_dir, "./iris.csv"),
                "delimiter": ",",
                "header": 0,
                "targets": ["variety"]
            },
            inputs = {},
            task_type = DatasetLoader
        )
        outputs = await tester.run()
        ds = outputs['dataset']

        vw = DatasetView(ds)
        dic = vw.to_dict()

        self.assertEqual(dic["type"], "dataset-view")
        self.assertEqual(
            dic["data"],
            ds.get_data().iloc[0:49, 0:4].to_dict('list')
        )

        self.assertEqual(
            dic["targets"],
            ds.get_targets().iloc[0:49, :].to_dict('list')
        )
