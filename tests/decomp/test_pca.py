
import os
import numpy

from gws_gaia import Dataset, DatasetLoader
from gws_gaia import PCATrainer, PCATransformer
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_pca(self):
        self.print("Principal Component Analysis (PCA)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./iris.csv"), 
            delimiter=",", 
            header=0, 
            targets=['variety']
        )

        # run trainer
        tester = TaskTester(
            params = {'nb_components': 2},
            inputs = {'dataset': dataset},
            task_type = PCATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        vm = trainer_result.view_transformed_data_as_table()
        dic = vm.to_dict()
        self.assertEqual(dic["type"], "table-view")

        vm = trainer_result.view_variance_as_table()
        dic = vm.to_dict()
        self.assertEqual(dic["type"], "table-view")
        self.assertTrue(numpy.all(numpy.isclose(dic["data"]["ExplainedVariance"], [0.92461, 0.053066], atol=1e-3)))

        vm = trainer_result.view_scores_as_2d_plot()
        dic = vm.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")
        self.assertTrue(numpy.all(numpy.isclose(dic["series"][0]["data"]["x"][0:3], [-2.6841, -2.714, -2.8889], atol=1e-3)))

        # run transformer
        tester = TaskTester(
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
