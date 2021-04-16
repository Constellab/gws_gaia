
import os
import asyncio
import unittest

from gaia.recurrentlayers import SimpleRNN, LSTM, GRU
from gaia.data import InputConverter
from gws.model import Protocol, Experiment, Study
#from gws.settings import Settings
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        SimpleRNN.drop_table()
        LSTM.drop_table()
        GRU.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()

    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        SimpleRNN.drop_table()
        LSTM.drop_table()
        GRU.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
        p1 = InputConverter()
        p2 = SimpleRNN()
        p3 = LSTM()
        p4 = GRU()

        proto = Protocol(
            processes = {
                'p1' : p1,
                'p2' : p2,
                'p3' : p3,
                'p4' : p4
            },
            connectors = [
                p1>>'result' | p2<<'tensor',
                p1>>'result' | p3<<'tensor',
                p1>>'result' | p4<<'tensor'
            ]
        )

        p1.set_param('input_shape', [3, 3])
        p2.set_param('units', 32)    
        p2.set_param('activation_type', 'tanh')    
        p2.set_param('use_bias', True)    
        p3.set_param('units', 32)    
        p3.set_param('activation_type', 'tanh')    
        p3.set_param('recurrent_activation_type', 'sigmoid')    
        p3.set_param('use_bias', True)    
        p4.set_param('units', 32)    
        p4.set_param('activation_type', 'tanh')    
        p4.set_param('recurrent_activation_type', 'sigmoid')    
        p4.set_param('use_bias', True)    

        def _end(*args, **kwargs):
            r1 = p2.output['result']
            r2 = p3.output['result']
            r3 = p4.output['result']

            print(r1)
            print(r2)
            print(r3)

        proto.on_end(_end)
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        
        asyncio.run( e.run() )                

        
        
