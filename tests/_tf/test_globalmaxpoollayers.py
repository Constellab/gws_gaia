import os
import asyncio


from gws_gaia.tf import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskRunner

class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        self.print("Global Max pooling operation for 1D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run GlobalMaxPooling1D
        tester = TaskRunner(
            params = {},
            inputs = {'tensor': in1},
            task_type = GlobalMaxPooling1D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        self.print("Global Max pooling operation for 2D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run GlobalMaxPooling2D
        tester = TaskRunner(
            params = {},
            inputs = {'tensor': in2},
            task_type = GlobalMaxPooling2D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        self.print("Global Max pooling operation for 3D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run GlobalMaxPooling3D
        tester = TaskRunner(
            params = {},
            inputs = {'tensor': in3},
            task_type = GlobalMaxPooling3D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)
