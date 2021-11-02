import os
import asyncio


from gws_gaia import Dataset
from gws_gaia.tf import ImporterPKL, Preprocessor, AdhocExtractor
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Importing and Preprocessing of PKL files")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        
        #run importerpkl
        tester = TaskTester(
           params = {'file_path': os.path.join(test_dir, "./mnist.pkl")},
           inputs = {},
           task_type = ImporterPKL
         )
        outputs = await tester.run()
        dataset = outputs['result']
        
        # run preprocessor
        tester = TaskTester(
            params = {'number_classes': 10},
            inputs = {'data': dataset},
            task_type = Preprocessor
        )
        outputs = await tester.run()
        preprocessor_result = outputs['result']

        # run extractor
        tester = TaskTester(
            params = {},
            inputs = {'data': dataset},
            task_type = AdhocExtractor
        )
        outputs = await tester.run()
        extractor_result = outputs['result']

        print(preprocessor_result)
        print(extractor_result)
