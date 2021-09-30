import os
import asyncio


from gws_gaia.tf import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from gws_gaia.tf import InputConverter
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):
    
    async def test_process_1D(self):
        self.print("Global Max pooling operation for 1D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in1 = outputs['result']

        # run GlobalMaxPooling1D
        tester = TaskTester(
            params = {},
            inputs = {'tensor': in1},
            task_type = GlobalMaxPooling1D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_2D(self):
        self.print("Global Max pooling operation for 2D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in2 = outputs['result']

        # run GlobalMaxPooling2D
        tester = TaskTester(
            params = {},
            inputs = {'tensor': in2},
            task_type = GlobalMaxPooling2D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

    async def test_process_3D(self):
        self.print("Global Max pooling operation for 3D data")
        # run InputConverter
        tester = TaskTester(
            params = {'input_shape': [None, 3, 3, 3]},
            inputs = {},
            task_type = InputConverter
        )
        outputs = await tester.run()
        in3 = outputs['result']

        # run GlobalMaxPooling3D
        tester = TaskTester(
            params = {},
            inputs = {'tensor': in3},
            task_type = GlobalMaxPooling3D
        )
        outputs = await tester.run()
        result = outputs['result']
        print(result)

# class TestTrainer(BaseTestCase):

        
#     async def test_process(self):
#         self.print("Global max pooling operation for 1D data")
#         p1 = InputConverter()
#         p2 = InputConverter()
#         p3 = InputConverter()
#         p4 = GlobalMaxPooling1D()
#         p5 = GlobalMaxPooling2D()
#         p6 = GlobalMaxPooling3D()
        
#         proto = Protocol(
#             processes = {
#                 'p1' : p1,
#                 'p2' : p2,
#                 'p3' : p3,
#                 'p4' : p4,
#                 'p5' : p5,
#                 'p6' : p6
#             },
#             connectors = [
#         p1>>'result' | p4<<'tensor',
#         p2>>'result' | p5<<'tensor',
#         p3>>'result' | p6<<'tensor'
#             ]
#         )

#         p1.set_param('input_shape', [None, 3])
#         p2.set_param('input_shape', [None, 3, 3])
#         p3.set_param('input_shape', [None, 3, 3 ,3])

#         experiment: Experiment = Experiment(
#             protocol=proto, study=GTest.study, user=GTest.user)
#         experiment.save()
#         experiment = await ExperimentService.run_experiment(
#             experiment=experiment, user=GTest.user)                

#         r1 = p4.output['result']
#         r2 = p5.output['result']
#         r3 = p6.output['result']            

#         print(r1)
#         print(r2)
#         print(r3)


        
        
