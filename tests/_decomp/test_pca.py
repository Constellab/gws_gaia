import numpy
from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import PCATrainer, PCATransformer


class TestTrainer(BaseTestCase):

    async def test_pca(self):
        self.print("Principal Component Analysis (PCA)")
        dataset = DataProvider.get_iris_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'dataset': dataset},
            task_type=PCATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # ---------------------------------------------------------------------
        # test views
        tester = ViewTester(
            view=trainer_result.view_transformed_data_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        # -----------------------------------------
        tester = ViewTester(
            view=trainer_result.view_variance_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")
        self.assertTrue(numpy.all(numpy.isclose(dic["data"]["ExplainedVariance"], [0.92461, 0.053066], atol=1e-3)))

        # -----------------------------------------
        tester = ViewTester(
            view=trainer_result.view_variance_as_barplot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "bar-plot-view")
        self.assertEqual(dic["data"]["series"][0]["data"]["x"], ['PC1', 'PC2'])
        self.assertTrue(numpy.all(
            numpy.isclose(
                dic["data"]["series"][0]["data"]["y"],
                [0.9246187232017271, 0.053066483117067804],
                atol=1e-3)))

        # -----------------------------------------
        tester = ViewTester(
            view=trainer_result.view_scores_as_2d_plot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")
        self.assertTrue(numpy.all(
            numpy.isclose(
                dic["data"]["series"][0]["data"]["x"][0:3],
                [-2.6841, -2.714, -2.8889],
                atol=1e-3)))

        # -----------------------------------------
        tester = ViewTester(
            view=trainer_result.view_scores_as_2d_plot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")

        # --------------------------------------------------------------------
        # run transformer
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=PCATransformer
        )
        outputs = await tester.run()
        transformer_result = outputs['result']

        print(trainer_result)
        print(transformer_result)
