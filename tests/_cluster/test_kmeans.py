from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import KMeansPredictor, KMeansTrainer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("K-means clustering")
        dataset = DataProvider.get_iris_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_clusters': 3},
            inputs={'dataset': dataset},
            task_type=KMeansTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # # ---------------------------------------------------------------------
        # # test views
        # tester = ViewTester(
        #     view=trainer_result.view_labels_as_table({})
        # )
        # dic = tester.to_dict()
        # self.assertEqual(dic["type"], "table-view")

        # tester = ViewTester(
        #     view=trainer_result.view_labels_as_2d_plot({})
        # )
        # dic = tester.to_dict()
        # self.assertEqual(dic["type"], "scatter-plot-2d-view")
        
        # ---------------------------------------------------------------------
        # run predictor
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=KMeansPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']
        # ---------------------------------------------------------------------

        print(trainer_result)
        print(predictor_result)
