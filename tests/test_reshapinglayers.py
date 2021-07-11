
import os
import asyncio
import unittest

from gaia.reshapinglayers import Flatten
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
        p1 = InputConverter()
        p2 = Flatten()

        proto = Protocol(
            processes = {
                'p1' : p1,
                'p2' : p2
            },
            connectors = [
                p1>>'result' | p2<<'tensor'
            ]
        )

        p1.set_param('input_shape', [3, 3, 3])    

        def _end(*args, **kwargs):
            r = p2.output['result']

            print(r)

        
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_end)
        asyncio.run( e.run() )    
        
        
