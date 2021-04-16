
import os
import asyncio
import unittest

from gaia.corelayers import Dense, Activation, Embedding, Masking
from gaia.data import InputConverter
from gws.model import Protocol, Experiment, Study
#from gws.settings import Settings
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Dense.drop_table()
        Activation.drop_table()
        Embedding.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        Dense.drop_table()
        Activation.drop_table()
        Embedding.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
        p1 = InputConverter()
        p2 = Dense()
        p3 = Activation()
        p4 = Embedding()
        p5 = Masking()

        proto = Protocol(
            processes = {
                'p1' : p1,
                'p2' : p2,
                'p3' : p3,
                'p4' : p4,
                'p5' : p5
            },
            connectors = [
                p1>>'result' | p2<<'tensor',
                p1>>'result' | p3<<'tensor',
                p1>>'result' | p4<<'tensor',
                p1>>'result' | p5<<'tensor'
            ]
        )

        p1.set_param('input_shape', [None, 3, 3])    
        p2.set_param('units', 32)    
        p2.set_param('activation', 'relu')    
        p2.set_param('use_bias', True)    
        p3.set_param('activation_type', 'relu')
        p4.set_param('input_dimension', 1000)
        p4.set_param('output_dimension', 64)
        p4.set_param('input_length', 10)
        p5.set_param('mask_value', 0.0)

        def _end(*args, **kwargs):
            r2 = p2.output['result']
            r3 = p3.output['result']
            r4 = p4.output['result']
            r5 = p5.output['result']

            print(r2)
            print(r3)
            print(r4)
            print(r5)
            
        proto.on_end(_end)        
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
 
        asyncio.run( e.run() )        
        
