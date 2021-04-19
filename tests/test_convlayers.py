
import os
import asyncio
import unittest

from gaia.convlayers import Conv1D, Conv2D, Conv3D
from gaia.data import InputConverter
from gws.model import Protocol, Experiment, Study
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Conv1D.drop_table()
        Conv2D.drop_table()
        Conv3D.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        Conv1D.drop_table()
        Conv2D.drop_table()
        Conv3D.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
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

        def _end(*args, **kwargs):
            r1 = p4.output['result']
            r2 = p5.output['result']
            r3 = p6.output['result']

            print(r1)
            print(r2)
            print(r3)
     
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_end) 
        asyncio.run( e.run() )

        
        
