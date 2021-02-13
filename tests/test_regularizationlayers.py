
import os
import asyncio
import unittest

from gaia.regularizationlayers import Dropout
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
        Dropout.drop_table()

    def test_process(self):
        p1 = InputConverter()
        p2 = Dropout()

        proto = Protocol(
            processes = {
                'p1' : p1,
                'p2' : p2
            },
            connectors = [
                p1>>'result' | p2<<'tensor'
            ]
        )

        p1.set_param('input_shape', [None, 3, 3])    
        p2.set_param('rate', 0.5)    

        def _end(*args, **kwargs):
            r = p2.output['result']

            print(r)

        proto.on_end(_end)
        e = proto.create_experiment()
        
        asyncio.run( e.run() )                
        
        
