
import os
import asyncio


from gws_core import Dataset
from gws_gaia import QDATrainer, QDAPredictor
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Quadratic discriminant analysis")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./dataset3.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )

        # run trainer
        tester = TaskRunner(
            params = {'reg_param': 0},
            inputs = {'dataset': dataset},
            task_type = QDATrainer
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
            task_type = QDAPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
