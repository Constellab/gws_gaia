import os
import asyncio
import unittest

from gaia.globalmaxpoollayers import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from gaia.data import InputConverter
from gws.protocol import Protocol
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    def test_process(self):
        GTest.print("Global max pooling operation for 1D data")
        p1 = InputConverter()
        p2 = InputConverter()
        p3 = InputConverter()
        p4 = GlobalMaxPooling1D()
        p5 = GlobalMaxPooling2D()
        p6 = GlobalMaxPooling3D()
        
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



        
        
