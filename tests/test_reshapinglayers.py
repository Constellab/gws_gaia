
import os
import asyncio
import unittest

from gaia.reshapinglayers import Flatten
from gaia.data import InputConverter
from gws.model import Protocol, Experiment, Study
#from gws.settings import Settings
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        Flatten.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()

    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        Flatten.drop_table()
        Protocol.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
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

        proto.on_end(_end)
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        
        asyncio.run( e.run() )    
        
        
