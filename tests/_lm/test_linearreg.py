
import os
import asyncio


from gws_gaia import Dataset
from gws_gaia import LinearRegressionTrainer, LinearRegressionPredictor, LinearRegressionTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester, ViewTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear regression analysis")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./diabetes.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )

        # run trainer
        tester = TaskTester(
            params = {},
            inputs = {'dataset': dataset},
            task_type = LinearRegressionTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        tester = ViewTester(
            view = trainer_result.view_predictions_as_table()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view = trainer_result.view_predictions_as_2d_plot()
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = LinearRegressionPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = LinearRegressionTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
        print(tester_result)
