
import os
import asyncio
import unittest

from gaia.averagepoollayers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from gaia.data import InputConverter
from gws.model import Protocol, Study, Experiment, Job
#from gws.settings import Settings
from gws.unittest import GTest

class TestTrainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        AveragePooling1D.drop_table()
        AveragePooling2D.drop_table()
        AveragePooling3D.drop_table()
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        #Dataset.drop_table()
        AveragePooling1D.drop_table()
        AveragePooling2D.drop_table()
        AveragePooling3D.drop_table()
        Protocol.drop_table()
        Job.drop_table()
        Experiment.drop_table()
        Study.drop_table()
        
    def test_process(self):
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

        def _end(*args, **kwargs):
            r1 = p4.output['result']
            r2 = p5.output['result']
            r3 = p6.output['result']            

            print(r1)
            print(r2)
            print(r3)
            
        proto.on_end(_end)
        e = proto.create_experiment(study=GTest.study, user=GTest.user)
        
        asyncio.run( e.run() )                



        
        
