
import os
import asyncio


from gws_core import (Settings, GTest, ConfigParams, ExperimentService, 
                        BaseTestCase, IntParam, TaskTester)
from gws_gaia import Dataset, DatasetLoader
from gws_gaia import AgglomerativeClusteringTrainer

class TestTrainer(BaseTestCase):

    async def test_process(self):
        GTest.print("Agglomerative clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset1.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target1','target2']
        )

        tester = TaskTester(
            task_type = AgglomerativeClusteringTrainer,
            params={'nb_clusters': 2},
            inputs={'dataset': dataset}
        )
        outputs = await tester.run()
        r1 = outputs['result']
        
        print(r1)