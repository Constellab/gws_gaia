
import os
import asyncio


from gws_gaia.tf import Conv1D, Conv2D, Conv3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, BaseTestCase, TaskTester, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process_1D(self):
        self.print("Convolutional layers")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run AveragePooling1D
        tester = TaskTester(
            params = ConfigParams({
                'nb_filters': 32,
                'kernel_size': 3,
                'activation_type': 'relu'
            }),
            inputs = {'tensor': in1},
            task_type = Conv1D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        self.print("Average pooling operation for 2D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [3, 3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run AveragePooling2D
        tester = TaskTester(
            params = ConfigParams({
                'nb_filters': 32,
                'kernel_size': [3, 3],
                'activation_type': 'relu'
            }),
            inputs = {'tensor': in2},
            task_type = Conv2D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        self.print("Average pooling operation for 3D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [3, 3, 3, 3,]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run AveragePooling3D
        tester = TaskTester(
            params = {
                'nb_filters': 32,
                'kernel_size': [3, 3, 3],
                'activation_type': 'relu'
            },
            inputs = {'tensor': in3},
            task_type = Conv3D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)
