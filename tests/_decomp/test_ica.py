import os
import asyncio


from gws_core import Dataset, DatasetImporter
from gws_gaia import ICATrainer
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Independant Component Analysis (ICA)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        
        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./digits.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":[]
            })
        )
        
        # run trainer
        tester = TaskRunner(
            params = {'nb_components': 7},
            inputs = {'dataset': dataset},
            task_type = ICATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        print(trainer_result)
