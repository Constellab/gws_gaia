# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from gws_core import Protocol, Settings, GTest, BaseTestCase, TaskTester, protocol_decorator
from gws_gaia import Tuple
from gws_gaia.tf import (Conv2D, MaxPooling2D, Flatten, Dropout, Dense, 
                            ImporterPKL, Preprocessor, AdhocExtractor, 
                            Tensor, DeepModel, InputConverter, 
                            DeepModelerBuilder, DeepModelerCompiler, DeepModelerTrainer,
                            DeepModelerTester, DeepModelerPredictor)

@protocol_decorator("DeepMoldelTurorialProto", 
                    human_name="Deep Moldel Proto", 
                    short_description="Turorial: Deep moldel protocol")
class DeepMoldelTurorialProto(Protocol):
    pass