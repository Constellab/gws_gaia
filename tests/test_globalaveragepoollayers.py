import os
import asyncio


from gws_gaia.tf import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        self.print("Global average pooling operation for 1D data")
        p1 = InputConverter()
        p2 = InputConverter()
        p3 = InputConverter()
        p4 = GlobalAveragePooling1D()
        p5 = GlobalAveragePooling2D()
        p6 = GlobalAveragePooling3D()
        
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
            

        
        
