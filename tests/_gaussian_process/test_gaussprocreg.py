import os
import asyncio


from gws_core import Dataset
from gws_gaia import GaussianProcessRegressorTrainer, GaussianProcessRegressorPredictor
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian Process Regressor")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./dataset2.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )

        # run trainer
        tester = TaskRunner(
            params = {'alpha': 1e-10},
            inputs = {'dataset': dataset},
            task_type = GaussianProcessRegressorTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = GaussianProcessRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
