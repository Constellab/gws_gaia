
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia.tf import Conv1D, Conv2D, Conv3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, Protocol, Experiment, ExperimentService

class TestTrainer(IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    async def test_process(self):
        GTest.print("1D convolution layer")
        p1 = InputConverter()
        p2 = InputConverter()
        p3 = InputConverter()        
        p4 = Conv1D()
        p5 = Conv2D()
        p6 = Conv3D()

        proto = Protocol(
            processes = {
                "p1": p1,
                "p2": p2,
                "p3": p3,                
                "p4": p4,                
                "p5": p5,                
                "p6": p6                
            },
            connectors = [
                p1>>'result' | p4<<'tensor',
                p2>>'result' | p5<<'tensor',
                p3>>'result' | p6<<'tensor'                
            ]
        )
        
        p1.set_param('input_shape', [3, 3, 3])
        p2.set_param('input_shape', [3, 3, 3, 3])        
        p3.set_param('input_shape', [3, 3, 3, 3])
        p4.set_param('nb_filters', 32)
        p4.set_param('kernel_size', 3)
        p4.set_param('activation_type', 'relu')    
        p5.set_param('nb_filters', 32)
        p5.set_param('kernel_size', [3, 3])
        p5.set_param('activation_type', 'relu')
        p6.set_param('nb_filters', 32)
        p6.set_param('kernel_size', [3, 3, 3])
        p6.set_param('activation_type', 'relu')

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
    
