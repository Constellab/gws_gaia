
import os
import asyncio


from gws_gaia import Dataset
from gws_gaia import KMeansTrainer, KMeansPredictor
from gws_core import Settings, GTest, BaseTestCase, TaskTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("K-means clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./iris.csv")), 
            ConfigParams({
                "delimiter": ",", 
                "header":0, 
                "targets":['variety']
            })
        )

        # run trainer
        tester = TaskTester(
            params = {'nb_clusters': 2},
            inputs = {'dataset': dataset},
            task_type = KMeansTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = KMeansPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
