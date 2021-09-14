
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import DecisionTreeClassifierTrainer, DecisionTreeClassifierPredictor, DecisionTreeClassifierTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        GTest.print("Decision Tree Classifier")
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
            params = ConfigParams({'max_depth': 4}),
            inputs = TaskInputs({'dataset': dataset}),
            task = DecisionTreeClassifierTrainer()
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = DecisionTreeClassifierPredictor()
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = DecisionTreeClassifierTester()
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
