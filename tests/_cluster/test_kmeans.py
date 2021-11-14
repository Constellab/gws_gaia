
import os
import asyncio


from gws_gaia import Dataset
from gws_gaia import KMeansTrainer, KMeansPredictor
from gws_core import Settings, GTest, BaseTestCase, TaskTester, File, ConfigParams, ViewTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("K-means clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        #---------------------------------------------------------------------
        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./iris.csv")), 
            ConfigParams({
                "delimiter": ",", 
                "header":0, 
                "targets":['variety']
            })
        )
        #---------------------------------------------------------------------
        # run trainer
        tester = TaskTester(
            params = {'nb_clusters': 3},
            inputs = {'dataset': dataset},
            task_type = KMeansTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        params = ConfigParams()
        #---------------------------------------------------------------------
        # test views
        tester = ViewTester(
            view = trainer_result.view_labels_as_table(params)
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
        #---------------------------------------------------------------------
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
