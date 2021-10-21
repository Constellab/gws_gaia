
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import LinearRegressionTrainer, LinearRegressionPredictor, LinearRegressionTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear regression analysis")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./diabetes.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target']
        )

        # run trainer
        tester = TaskTester(
            params = {},
            inputs = {'dataset': dataset},
            task_type = LinearRegressionTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        vm = trainer_result.view_y_data_vs_y_predicted_as_table()
        dic = vm.to_dict()
        self.assertEqual(dic["type"], "table-view")

        vm = trainer_result.view_y_data_vs_y_predicted_as_2dplot()
        dic = vm.to_dict()
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
