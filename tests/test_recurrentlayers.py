import os
import asyncio


from gws_gaia.tf import SimpleRNN, LSTM, GRU
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):
    
    async def test_process(self):
        self.print("Recurrent layers")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run SimpleRNN
        tester = TaskTester(
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
        tester = TaskTester(
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
        tester = TaskTester(
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


# class TestTrainer(BaseTestCase):

        
#     async def test_process(self):
#         self.print("Long Short-Term Memory (LSTM) layer")
#         p1 = InputConverter()
#         p2 = SimpleRNN()
#         p3 = LSTM()
#         p4 = GRU()

#         proto = Protocol(
#             processes = {
#                 'p1' : p1,
#                 'p2' : p2,
#                 'p3' : p3,
#                 'p4' : p4
#             },
#             connectors = [
#                 p1>>'result' | p2<<'tensor',
#                 p1>>'result' | p3<<'tensor',
#                 p1>>'result' | p4<<'tensor'
#             ]
#         )

#         p1.set_param('input_shape', [3, 3])
#         p2.set_param('units', 32)    
#         p2.set_param('activation_type', 'tanh')    
#         p2.set_param('use_bias', True)    
#         p3.set_param('units', 32)    
#         p3.set_param('activation_type', 'tanh')    
#         p3.set_param('recurrent_activation_type', 'sigmoid')    
#         p3.set_param('use_bias', True)    
#         p4.set_param('units', 32)    
#         p4.set_param('activation_type', 'tanh')    
#         p4.set_param('recurrent_activation_type', 'sigmoid')    
#         p4.set_param('use_bias', True)    

#         experiment: Experiment = Experiment(
#             protocol=proto, study=GTest.study, user=GTest.user)
#         experiment.save()
#         experiment = await ExperimentService.run_experiment(
#             experiment=experiment, user=GTest.user)                

#         r1 = p2.output['result']
#         r2 = p3.output['result']
#         r3 = p4.output['result']

#         print(r1)
#         print(r2)
#         print(r3)
        
