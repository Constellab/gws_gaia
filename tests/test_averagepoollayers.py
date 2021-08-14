import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_gaia import Tuple
from gws_gaia.tf import InputConverter, AveragePooling1D, AveragePooling2D, AveragePooling3D
from gws_core import GTest, Protocol, Experiment, ExperimentService

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
        GTest.print("Average pooling operation for 1D data")
        p1 = InputConverter()
        p2 = InputConverter()
        p3 = InputConverter()
        p4 = AveragePooling1D()
        p5 = AveragePooling2D()
        p6 = AveragePooling3D()
        
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

        
        
