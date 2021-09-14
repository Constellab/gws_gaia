import os
import asyncio

from gws_gaia import Tuple
from gws_gaia.tf import InputConverter, AveragePooling1D, AveragePooling2D, AveragePooling3D
from gws_core import GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        GTest.print("Average pooling operation for 1D data")
        # run InputConverter
        tester = TaskTester(
            params = ConfigParams({'input_shape': [None, 3]}),
            inputs = TaskInputs(),
            task = InputConverter()
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run AveragePooling1D
        tester = TaskTester(
            params = ConfigParams({'pool_size': 2}),
            inputs = TaskInputs({'tensor': in1}),
            task = AveragePooling1D()
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        GTest.print("Average pooling operation for 2D data")
        # run InputConverter
        tester = TaskTester(
            params = ConfigParams({'input_shape': [None, 3, 3]}),
            inputs = TaskInputs(),
            task = InputConverter()
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run AveragePooling2D
        tester = TaskTester(
            params = ConfigParams({'pool_size': [2, 2]}),
            inputs = TaskInputs({'tensor': in2}),
            task = AveragePooling2D()
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        GTest.print("Average pooling operation for 3D data")
        # run InputConverter
        tester = TaskTester(
            params = ConfigParams({'input_shape': [None, 3, 3, 3]}),
            inputs = TaskInputs(),
            task = InputConverter()
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run AveragePooling3D
        tester = TaskTester(
            params = ConfigParams({'pool_size': [2, 2, 3]}),
            inputs = TaskInputs({'tensor': in3}),
            task = AveragePooling3D()
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)
