
import os
import asyncio
import unittest

from gaia.maxpoollayers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from gaia.data import InputConverter
from gws.model import Protocol
#from gws.settings import Settings

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        MaxPooling1D.drop_table()

    def test_process(self):
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

        def _end(*args, **kwargs):
            r1 = p4.output['result']
            r2 = p5.output['result']
            r3 = p6.output['result']            

            print(r1)
            print(r2)
            print(r3)
            
        proto.on_end(_end)
        e = proto.create_experiment()
        
        asyncio.run( e.run() )      
        
        