import os
from gws_core import (Settings, TaskTester, BaseTestCase)
from gws_gaia import Dataset, ExtendedTableView, DatasetLoader


class TestTExtentedTableView(BaseTestCase):

    async def test_extended_table_view(self,):
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

        vw = ExtendedTableView(ds)
        dic = vw.to_dict()

        self.assertEqual(dic["type"], "extended-table")
        self.assertEqual(
            dic["data"],
            ds.get_data().iloc[0:49, 0:3].to_dict('list')
        )

        self.assertEqual(
            dic["targets"],
            ds.get_targets().iloc[0:49, :].to_dict('list')
        )
