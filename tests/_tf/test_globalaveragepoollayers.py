import os
import asyncio


from gws_gaia.tf import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskRunner

class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        self.print("Global Average pooling operation for 1D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run GlobalAveragePooling1D
        tester = TaskRunner(
            params = {'pool_size': 2},
            inputs = {'tensor': in1},
            task_type = GlobalAveragePooling1D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        self.print("Global Average pooling operation for 2D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run GlobalAveragePooling2D
        tester = TaskRunner(
            params = {'pool_size': [2, 2]},
            inputs = {'tensor': in2},
            task_type = GlobalAveragePooling2D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        self.print("Global Average pooling operation for 3D data")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run GlobalAveragePooling3D
        tester = TaskRunner(
            params = {'pool_size': [2, 2, 2]},
            inputs = {'tensor': in3},
            task_type = GlobalAveragePooling3D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)
