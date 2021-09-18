# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from gws_core import (Protocol, Settings, GTest, BaseTestCase, 
                        TaskTester, protocol_decorator, ProcessSpec, ConfigParams)
from gws_gaia import GenericResult
from gws_gaia.tf import (Conv2D, MaxPooling2D, Flatten, Dropout, Dense, 
                            ImporterPKL, Preprocessor, AdhocExtractor, 
                            Tensor, DeepModel, InputConverter, 
                            DeepModelerBuilder, DeepModelerCompiler, DeepModelerTrainer,
                            DeepModelerTester, DeepModelerPredictor)

@protocol_decorator("DeepMoldelTurorialProto", 
                    human_name="Deep Moldel Proto", 
                    short_description="Turorial: Deep moldel protocol")
class DeepMoldelTurorialProto(Protocol):
    def configure_protocol(self, config_params: ConfigParams) -> None:
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        pkl_importer: ProcessSpec = self.add_process(ImporterPKL, 'pkl_importer')
        preprocessor: ProcessSpec = self.add_process(Preprocessor, 'preprocessor')
        input_converter: ProcessSpec = self.add_process(InputConverter, 'input_converter')
        conv_2d_1: ProcessSpec = self.add_process(Conv2D, 'conv_2d_1')
        max_pooling_2d_1: ProcessSpec = self.add_process(MaxPooling2D, 'max_pooling_2d_1')
        conv_2d_2: ProcessSpec = self.add_process(Conv2D, 'conv_2d_2')
        max_pooling_2d_2: ProcessSpec = self.add_process(MaxPooling2D, 'max_pooling_2d_2')
        flatten: ProcessSpec = self.add_process(Flatten, 'flatten')
        dropout: ProcessSpec = self.add_process(Dropout, 'dropout')
        dense: ProcessSpec = self.add_process(Dense, 'dense')
        deep_modeler_builder: ProcessSpec = self.add_process(DeepModelerBuilder, 'deep_modeler_builder')
        deep_modeler_compiler: ProcessSpec = self.add_process(DeepModelerCompiler, 'deep_modeler_compiler')
        deep_modeler_trainer: ProcessSpec = self.add_process(DeepModelerTrainer, 'deep_modeler_trainer')
        deep_modeler_tester: ProcessSpec = self.add_process(DeepModelerTester, 'deep_modeler_tester')
        ad_hoc_extractor: ProcessSpec = self.add_process(AdhocExtractor, 'ad_hoc_extractor')
        deep_modeler_predictor: ProcessSpec = self.add_process(DeepModelerPredictor, 'deep_modeler_predictor')

        pkl_importer.set_param("file_path", os.path.join(test_dir, "./mnist.pkl"))
        preprocessor.set_param('number_classes', 10)
        input_converter.set_param('input_shape', [28, 28, 1])
        conv_2d_1.set_param('nb_filters', 32)
        conv_2d_1.set_param('kernel_size', [3,3])
        conv_2d_1.set_param('activation_type', 'relu')    
        max_pooling_2d_1.set_param('pool_size', [2, 2])
        conv_2d_2.set_param('nb_filters', 64)
        conv_2d_2.set_param('kernel_size', [3,3])
        conv_2d_2.set_param('activation_type', 'relu')    
        max_pooling_2d_2.set_param('pool_size', [2, 2])
        dropout.set_param('rate', 0.5)
        dense.set_param('units', 10)
        dense.set_param('activation','softmax')
        deep_modeler_compiler.set_param('loss', 'categorical_crossentropy')
        deep_modeler_compiler.set_param('optimizer', 'adam')
        deep_modeler_compiler.set_param('metrics', 'accuracy')    
        deep_modeler_trainer.set_param('batch_size', 128)
        deep_modeler_trainer.set_param('epochs', 2)
        deep_modeler_trainer.set_param('validation_split', 0.1)    
        deep_modeler_tester.set_param('verbosity_mode', 1)    
        deep_modeler_predictor.set_param('verbosity_mode', 1)

        self.add_connectors([
            (pkl_importer>>'result', preprocessor<<'data'),
            (preprocessor>>'result', ad_hoc_extractor<<'data'),
            (input_converter>>'result', conv_2d_1<<'tensor'),
            (conv_2d_1>>'result', max_pooling_2d_1<<'tensor'),
            (max_pooling_2d_1>>'result', conv_2d_2<<'tensor'),
            (conv_2d_2>>'result', max_pooling_2d_2<<'tensor'),
            (max_pooling_2d_2>>'result', flatten<<'tensor'),
            (flatten>>'result', dropout<<'tensor'),
            (dropout>>'result', dense<<'tensor'),
            (dense>>'result', deep_modeler_builder<<'outputs'),
            (input_converter>>'result', deep_modeler_builder<<'inputs'),
            (deep_modeler_builder>>'result', deep_modeler_compiler<<'builded_model'),            
            (deep_modeler_compiler>>'result', deep_modeler_trainer<<'compiled_model'),
            (preprocessor>>'result', deep_modeler_trainer<<'dataset'),
            (deep_modeler_trainer>>'result', deep_modeler_tester<<'trained_model'),
            (preprocessor>>'result', deep_modeler_tester<<'dataset'),
            (deep_modeler_trainer>>'result', deep_modeler_predictor<<'trained_model'),
            (ad_hoc_extractor>>'result', deep_modeler_predictor<<'dataset')
        ])