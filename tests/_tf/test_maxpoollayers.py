import os
import asyncio


from gws_gaia.tf import MaxPooling1D, MaxPooling2D, MaxPooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester


class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        self.print("Max pooling operation for 1D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run MaxPooling1D
        tester = TaskTester(
            params = {'pool_size': 2},
            inputs = {'tensor': in1},
            task_type = MaxPooling1D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        self.print("Max pooling operation for 2D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run MaxPooling2D
        tester = TaskTester(
            params = {'pool_size': [2, 2]},
            inputs = {'tensor': in2},
            task_type = MaxPooling2D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        self.print("Max pooling operation for 3D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run MaxPooling3D
        tester = TaskTester(
            params = {'pool_size': [2, 2, 2]},
            inputs = {'tensor': in3},
            task_type = MaxPooling3D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)
