import os
import asyncio


from gws_gaia.tf import MaxPooling1D, MaxPooling2D, MaxPooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        GTest.print("Max pooling operation for 1D data")
        p1 = InputConverter()
        p2 = InputConverter()
        p3 = InputConverter()
        p4 = MaxPooling1D()
        p5 = MaxPooling2D()
        p6 = MaxPooling3D()

        proto = Protocol(
            processes = {
                'p1' : p1,
                'p2' : p2,
                'p3' : p3,
                'p4' : p4,
                'p5' : p5,
                'p6' : p6
            },
            connectors = [
                p1>>'result' | p4<<'tensor',
                p2>>'result' | p5<<'tensor',
                p3>>'result' | p6<<'tensor'
            ]
        )

        p1.set_param('input_shape', [None, 3])
        p2.set_param('input_shape', [None, 3, 3])
        p3.set_param('input_shape', [None, 3, 3 ,3])
        p4.set_param('pool_size', 2)
        p5.set_param('pool_size', [2, 2])
        p6.set_param('pool_size', [2, 2, 2]) 
        
        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)      
        
        r1 = p4.output['result']
        r2 = p5.output['result']
        r3 = p6.output['result']            

        print(r1)
        print(r2)
        print(r3)
        
