
import os
import asyncio


from gws_gaia.tf import Dense, Activation, Embedding, Masking
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        GTest.print("Neural network layers")
        # run InputConverter
        tester = TaskTester(
            params = ConfigParams({'input_shape': [3, 3, 3]}),
            inputs = TaskInputs(),
            task = InputConverter()
        )
        outputs = await tester.run()
        result = outputs['result']

        # run Dense
        tester = TaskTester(
            params = ConfigParams({
                'units': 32,
                'activation': 'relu',   
                'use_bias': True
            }),
            inputs = TaskInputs({'tensor': result}),
            task = Dense()
        )
        outputs = await tester.run()
        result = outputs['result']

        # run Activation
        tester = TaskTester(
            params = ConfigParams({
                'activation_type': 'relu'
            }),
            inputs = TaskInputs({'tensor': result}),
            task = Activation()
        )
        outputs = await tester.run()
        result = outputs['result']

        # run Embedding
        tester = TaskTester(
            params = ConfigParams({
                'input_dimension': 1000,
                'output_dimension': 64,
                'input_length': 10
            }),
            inputs = TaskInputs({'tensor': result}),
            task = Embedding()
        )
        outputs = await tester.run()
        result = outputs['result']

        # run Masking
        tester = TaskTester(
            params = ConfigParams({
                'mask_value': 0.0
            }),
            inputs = TaskInputs({'tensor': result}),
            task = Masking()
        )
        outputs = await tester.run()
        result = outputs['result']

        print(result)

    # async def test_process(self):
    #     GTest.print("Densely connected Neural Network layer")
    #     p1 = InputConverter()
    #     p2 = Dense()
    #     p3 = Activation()
    #     p4 = Embedding()
    #     p5 = Masking()

    #     proto = Protocol(
    #         processes = {
    #             'p1' : p1,
    #             'p2' : p2,
    #             'p3' : p3,
    #             'p4' : p4,
    #             'p5' : p5
    #         },
    #         connectors = [
    #             p1>>'result' | p2<<'tensor',
    #             p1>>'result' | p3<<'tensor',
    #             p1>>'result' | p4<<'tensor',
    #             p1>>'result' | p5<<'tensor'
    #         ]
    #     )

    #     p1.set_param('input_shape', [None, 3, 3])    
    #     p2.set_param('units', 32)    
    #     p2.set_param('activation', 'relu')    
    #     p2.set_param('use_bias', True)    
    #     p3.set_param('activation_type', 'relu')
    #     p4.set_param('input_dimension', 1000)
    #     p4.set_param('output_dimension', 64)
    #     p4.set_param('input_length', 10)
    #     p5.set_param('mask_value', 0.0)
       
    #     experiment: Experiment = Experiment(
    #         protocol=proto, study=GTest.study, user=GTest.user)
    #     experiment.save()
    #     experiment = await ExperimentService.run_experiment(
    #         experiment=experiment, user=GTest.user)        
        
    #     r2 = p2.output['result']
    #     r3 = p3.output['result']
    #     r4 = p4.output['result']
    #     r5 = p5.output['result']

    #     print(r2)
    #     print(r3)
    #     print(r4)
    #     print(r5)