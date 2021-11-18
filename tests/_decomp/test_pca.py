
import os
import numpy

from gws_gaia import Dataset, DatasetImporter
from gws_gaia import PCATrainer, PCATransformer
from gws_core import Settings, GTest, BaseTestCase, TaskRunner, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_pca(self):
        self.print("Principal Component Analysis (PCA)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #---------------------------------------------------------------------
        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./iris.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['variety']
            })
        )

        #---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params = {'nb_components': 2},
            inputs = {'dataset': dataset},
            task_type = PCATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        #---------------------------------------------------------------------
        # test views
        tester = ViewTester(
            view = trainer_result.view_transformed_data_as_table()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        #-----------------------------------------
        tester = ViewTester(
            view = trainer_result.view_variance_as_table()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
        self.assertTrue(numpy.all(numpy.isclose(dic["data"]["ExplainedVariance"], [0.92461, 0.053066], atol=1e-3)))

        #-----------------------------------------
        tester = ViewTester(
            view = trainer_result.view_variance_as_barplot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "bar-plot-view")
        #self.assertTrue(numpy.all(numpy.isclose(dic["data"]["ExplainedVariance"], [0.92461, 0.053066], atol=1e-3)))

        #-----------------------------------------
        #vm = trainer_result.view_scores_as_2d_plot()
        #dic = vm.to_dict()
        tester = ViewTester(
            view = trainer_result.view_scores_as_2d_plot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")
        self.assertTrue(numpy.all(numpy.isclose(dic["data"][0]["data"]["x"][0:3], [-2.6841, -2.714, -2.8889], atol=1e-3)))
        
        #-----------------------------------------
        tester = ViewTester(
            view = trainer_result.view_scores_as_2d_plot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")

        #--------------------------------------------------------------------
        # run transformer
        tester = TaskRunner(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = PCATransformer
        )
        outputs = await tester.run()
        transformer_result = outputs['result']
       
        print(trainer_result)
        print(transformer_result)
