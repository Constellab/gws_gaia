
import os
import asyncio


from gws_gaia.tf import Flatten
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        self.print("Flatten layer")
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

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)    
        
        r = p2.output['result']
        print(r)
