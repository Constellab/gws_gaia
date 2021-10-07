
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import LDATrainer, LDATester, LDAPredictor, LDATransformer
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear discriminant analysis classifier")
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
            params = {
                'solver': 'svd',
                'nb_components': 2
            },
            inputs = {'dataset': dataset},
            task_type = LDATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = LDAPredictor
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
            task_type = LDATester
        )
        outputs = await tester.run()
        tester_result = outputs['result']
       
        # run transformer
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = LDATransformer
        )
        outputs = await tester.run()
        transformer_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
        print(tester_result)
        print(transformer_result)
