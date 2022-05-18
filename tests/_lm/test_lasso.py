from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner, ViewTester)
from gws_gaia import LassoPredictor, LassoTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Lasso regression")
        dataset = DataProvider.get_diabetes_dataset()
        #--------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'alpha': 1},
            inputs={'dataset': dataset},
            task_type=LassoTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        # #--------------------------------------------------------------------
        # # run views
        # tester = ViewTester(
        #     view=trainer_result.view_predictions_as_table({})
        # )
        # dic = tester.to_dict()
        # self.assertEqual(dic["type"], "table-view")

        # tester = ViewTester(
        #     view=trainer_result.view_predictions_as_2d_plot({})
        # )
        # dic = tester.to_dict()
        # self.assertEqual(dic["type"], "scatter-plot-2d-view")
        #-----------------------------------------------------------------
        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=LassoPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']
        #--------------------------------------------------------------------

        print(trainer_result)
        print(predictor_result)
