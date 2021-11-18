import os
import asyncio


from gws_gaia.tf import Dropout
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskRunner

class TestTrainer(BaseTestCase):
    
    async def test_process(self):
        self.print("Dropout layer")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [None, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run Dropout
        tester = TaskRunner(
            params = {
                'rate': 0.5
            },
            inputs = {'tensor': in1},
            task_type = Dropout
        )
        outputs = await tester.run()
        result = outputs['result']

        print(result)
