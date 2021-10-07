
import os
import asyncio


from gws_gaia.tf import Flatten
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):
    
    async def test_process(self):
        self.print("Flatten layer")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run Flatten
        tester = TaskTester(
            params = {},
            inputs = {'tensor': in1},
            task_type = Flatten
        )
        outputs = await tester.run()
        result = outputs['result']

        print(result)
