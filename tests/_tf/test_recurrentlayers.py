import os
import asyncio


from gws_gaia.tf import SimpleRNN, LSTM, GRU
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskRunner

class TestTrainer(BaseTestCase):
    
    async def test_process(self):
        self.print("Recurrent layers")
        # run InputConverter
        tester = TaskRunner(
            params = {'input_shape': [3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run SimpleRNN
        tester = TaskRunner(
            params = {
                'units': 32,
                'activation_type': 'tanh',
                'use_bias': True
            },
            inputs = {'tensor': in1},
            task_type = SimpleRNN
        )
        outputs = await tester.run()
        simplernn_result = outputs['result']

        # run LSTM
        tester = TaskRunner(
            params = {
                'units': 32,
                'activation_type': 'tanh',
                'recurrent_activation_type': 'sigmoid',
                'use_bias': True
            },
            inputs = {'tensor': in1},
            task_type = LSTM
        )
        outputs = await tester.run()
        lstm_result = outputs['result']

        # run GRU
        tester = TaskRunner(
            params = {
                'units': 32,
                'activation_type': 'tanh',
                'recurrent_activation_type': 'sigmoid',
                'use_bias': True
            },
            inputs = {'tensor': in1},
            task_type = GRU
        )
        outputs = await tester.run()
        gru_result = outputs['result']

        print(simplernn_result)
        print(lstm_result)
        print(gru_result)
