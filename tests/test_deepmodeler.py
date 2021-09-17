import os
import asyncio


from gws_core import (Protocol, Settings, GTest, BaseTestCase, 
                        ConfigParams, ProcessSpec, protocol_decorator,
                        IExperiment)
from gws_gaia import Tuple
from gws_gaia.tf import (Conv2D, MaxPooling2D, Flatten, Dropout, Dense, 
                            ImporterPKL, Preprocessor, AdhocExtractor, 
                            Tensor, DeepModel, InputConverter, 
                            DeepModelerBuilder, DeepModelerCompiler, DeepModelerTrainer,
                            DeepModelerTester, DeepModelerPredictor)

@protocol_decorator("DeepMoldelTurorialProto")
class DeepMoldelTurorialProto(Protocol):
    def configure_protocol(self, config_params: ConfigParams) -> None:
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        p0: ProcessSpec = self.add_process(ImporterPKL, 'p0')
        p1: ProcessSpec = self.add_process(Preprocessor, 'p1')
        p2: ProcessSpec = self.add_process(InputConverter, 'p2')
        p3: ProcessSpec = self.add_process(Conv2D, 'p3')
        p4: ProcessSpec = self.add_process(MaxPooling2D, 'p4')
        p5: ProcessSpec = self.add_process(Conv2D, 'p5')
        p6: ProcessSpec = self.add_process(MaxPooling2D, 'p6')
        p7: ProcessSpec = self.add_process(Flatten, 'p7')
        p8: ProcessSpec = self.add_process(Dropout, 'p8')
        p9: ProcessSpec = self.add_process(Dense, 'p9')
        p10: ProcessSpec = self.add_process(DeepModelerBuilder, 'p10')
        p11: ProcessSpec = self.add_process(DeepModelerCompiler, 'p11')
        p12: ProcessSpec = self.add_process(DeepModelerTrainer, 'p12')
        p13: ProcessSpec = self.add_process(DeepModelerTester, 'p13')
        p14: ProcessSpec = self.add_process(AdhocExtractor, 'p14')
        p15: ProcessSpec = self.add_process(DeepModelerPredictor, 'p15')

        p0.set_param("file_path", os.path.join(test_dir, "./mnist.pkl"))
        p1.set_param('number_classes', 10)
        p2.set_param('input_shape', [28, 28, 1])
        p3.set_param('nb_filters', 32)
        p3.set_param('kernel_size', [3,3])
        p3.set_param('activation_type', 'relu')    
        p4.set_param('pool_size', [2, 2])
        p5.set_param('nb_filters', 64)
        p5.set_param('kernel_size', [3,3])
        p5.set_param('activation_type', 'relu')    
        p6.set_param('pool_size', [2, 2])
        p8.set_param('rate', 0.5)
        p9.set_param('units', 10)
        p9.set_param('activation','softmax')
        p11.set_param('loss', 'categorical_crossentropy')
        p11.set_param('optimizer', 'adam')
        p11.set_param('metrics', 'accuracy')    
        p12.set_param('batch_size', 128)
        p12.set_param('epochs', 2)
        p12.set_param('validation_split', 0.1)    
        p13.set_param('verbosity_mode', 1)    
        p15.set_param('verbosity_mode', 1)

        self.add_connectors([
            (p0>>'result', p1<<'data'),
            (p1>>'result', p14<<'data'),
            (p2>>'result', p3<<'tensor'),
            (p3>>'result', p4<<'tensor'),
            (p4>>'result', p5<<'tensor'),
            (p5>>'result', p6<<'tensor'),
            (p6>>'result', p7<<'tensor'),
            (p7>>'result', p8<<'tensor'),
            (p8>>'result', p9<<'tensor'),
            (p9>>'result', p10<<'outputs'),
            (p2>>'result', p10<<'inputs'),
            (p10>>'result', p11<<'builded_model'),            
            (p11>>'result', p12<<'compiled_model'),
            (p1>>'result', p12<<'dataset'),
            (p12>>'result', p13<<'trained_model'),
            (p1>>'result', p13<<'dataset'),
            (p12>>'result', p15<<'trained_model'),
            (p14>>'result', p15<<'dataset')
        ])

class TestDeepModel(BaseTestCase):

    async def test_deep_model(self):
        experiment: IExperiment = IExperiment( DeepMoldelTurorialProto )
        await experiment.run()


# class TestTrainer(BaseTestCase):

        
#     async def test_process(self): 
#         GTest.print("Deep modeler")       
#         settings = Settings.retrieve()
#         test_dir = settings.get_variable("gws_gaia:testdata_dir")

#         p0 = ImporterPKL()
#         p1 = Preprocessor()

#         p2 = InputConverter()
#         p3 = Conv2D()
#         p4 = MaxPooling2D()
#         p5 = Conv2D()
#         p6 = MaxPooling2D()
#         p7 = Flatten()
#         p8 = Dropout()
#         p9 = Dense()

#         p10 = DeepModelerBuilder()
#         p11 = DeepModelerCompiler()
#         p12 = DeepModelerTrainer()
#         p13 = DeepModelerTester()
#         p14 = AdhocExtractor()
#         p15 = DeepModelerPredictor()

#         proto = Protocol(
#             processes = {
#                 "p0": p0,
#                 "p1": p1,
#                 "p2": p2,
#                 "p3": p3,
#                 "p4": p4,
#                 "p5": p5,
#                 "p6": p6,
#                 "p7": p7,
#                 "p8": p8,
#                 "p9": p9,
#                 "p10": p10,
#                 "p11": p11,
#                 "p12": p12,
#                 "p13": p13,                
#                 "p14": p14,
#                 "p15": p15               
#             },
#             connectors = [
#                 p0>>'result' | p1<<'data',
#                 p1>>'result' | p14<<'data',
#                 p2>>'result' | p3<<'tensor',
#                 p3>>'result' | p4<<'tensor',
#                 p4>>'result' | p5<<'tensor',
#                 p5>>'result' | p6<<'tensor',
#                 p6>>'result' | p7<<'tensor',
#                 p7>>'result' | p8<<'tensor',
#                 p8>>'result' | p9<<'tensor',
#                 p9>>'result' | p10<<'outputs',
#                 p2>>'result' | p10<<'inputs',
#                 p10>>'result' | p11<<'builded_model',            
#                 p11>>'result' | p12<<'compiled_model',
#                 p1>>'result' | p12<<'dataset',
#                 p12>>'result' | p13<<'trained_model',
#                 p1>>'result' | p13<<'dataset',
#                 p12>>'result' | p15<<'trained_model',
#                 p14>>'result' | p15<<'dataset'
#             ]
#         )
        
#         p0.set_param("file_path", os.path.join(test_dir, "./mnist.pkl"))
#         p1.set_param('number_classes', 10)
#         p2.set_param('input_shape', [28, 28, 1])
#         p3.set_param('nb_filters', 32)
#         p3.set_param('kernel_size', [3,3])
#         p3.set_param('activation_type', 'relu')    
#         p4.set_param('pool_size', [2, 2])
#         p5.set_param('nb_filters', 64)
#         p5.set_param('kernel_size', [3,3])
#         p5.set_param('activation_type', 'relu')    
#         p6.set_param('pool_size', [2, 2])
#         p8.set_param('rate', 0.5)
#         p9.set_param('units', 10)
#         p9.set_param('activation','softmax')
#         p11.set_param('loss', 'categorical_crossentropy')
#         p11.set_param('optimizer', 'adam')
#         p11.set_param('metrics', 'accuracy')    
#         p12.set_param('batch_size', 128)
#         p12.set_param('epochs', 2)
#         p12.set_param('validation_split', 0.1)    
#         p13.set_param('verbosity_mode', 1)    
#         p15.set_param('verbosity_mode', 1)    

#         experiment: Experiment = Experiment(
#             protocol=proto, study=GTest.study, user=GTest.user)
#         experiment.save()
#         experiment = await ExperimentService.run_experiment(
#             experiment=experiment, user=GTest.user)
 
#         r1 = p12.output['result']
#         r1 = p15.output['result']
        
        
