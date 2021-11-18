
import os
import numpy

from gws_gaia import Dataset, DatasetImporter
from gws_gaia import PLSTrainer, PLSPredictor, PLSTransformer
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Partial Least Squares (PLS) regression")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        #---------------------------------------------------------------------
        #import data
        # dataset = Dataset.import_from_path(
        #     File(path=os.path.join(test_dir, "./dataset1.csv")),  
        #     ConfigParams({
        #         "delimiter":",", 
        #         "header":0, 
        #         "targets":['target1', 'target2']
        #     })
        # )
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./diabetes.csv")),  
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )

        #---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params = {'nb_components': 2},
            inputs = {'dataset': dataset},
            task_type = PLSTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        #---------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view = trainer_result.view_transformed_data_as_table()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        #------------------------------------------
        tester = ViewTester(
            view = trainer_result.view_scores_as_2d_plot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")
        #self.assertTrue(numpy.all(numpy.isclose(dic["data"][0]["data"]["x"], [-1.3970, -1.1967, 0.5603, 2.0334], atol=1e-3)))

        #------------------------------------------
        tester = ViewTester(
            view = trainer_result.view_predictions_as_table()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        #------------------------------------------
        tester = ViewTester(
            view = trainer_result.view_predictions_as_2d_plot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")

        #---------------------------------------------------------------------
        # run predictior
        tester = TaskRunner(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = PLSPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        #---------------------------------------------------------------------
        # run tester
        tester = TaskRunner(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = PLSTransformer
        )
        outputs = await tester.run()
        tester_result = outputs['result']
       
        print(trainer_result)
        print(predictor_result)
        print(tester_result)
