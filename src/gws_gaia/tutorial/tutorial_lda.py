# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from gws_core import (Protocol, Study, User, Experiment, 
                        protocol_decorator, ProcessSpec, 
                        ConfigParams, Settings)

from gws_gaia import Tuple
from gws_gaia import Dataset, DatasetLoader
from gws_gaia import  LDATrainer, LDAPredictor, LDATester, LDATransformer
from gws_gaia import PCATrainer, PCATransformer

@protocol_decorator("LDATutorialProto", 
                    human_name="LDA Proto", 
                    short_description="Tutorial: short LDA and PCA protocol")
class LDATutorialProto(Protocol):
    
    def configure_protocol(self, config_params: ConfigParams) -> None:
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        importer: ProcessSpec = self.add_process(DatasetLoader, 'importer')
        lda_trainer: ProcessSpec = self.add_process(LDATrainer, 'lda_trainer')
        lda_pred: ProcessSpec = self.add_process(LDAPredictor, 'lda_pred')
        lda_tester: ProcessSpec = self.add_process(LDATester, 'lda_tester')
        lda_trans: ProcessSpec = self.add_process(LDATransformer, 'lda_trans')
        pca_trainer: ProcessSpec = self.add_process(PCATrainer, 'pca_trainer')
        pca_trans: ProcessSpec = self.add_process(PCATransformer, 'pca_trans')

        data_file = os.path.join(test_dir, "./iris.csv")
        importer.set_param("delimiter", ",")
        importer.set_param("header", 0)
        importer.set_param('targets', ['variety']) 
        importer.set_param("file_path", data_file)
        lda_trainer.set_param('nb_components', 2)
        pca_trainer.set_param('nb_components', 2)

        self.add_connectors([
            (importer>>'dataset', pca_trainer<<'dataset'),                
            (importer>>'dataset', pca_trans<<'dataset'),                
            (pca_trainer>>'result', pca_trans<<'learned_model'),                
            (importer>>'dataset', lda_trainer<<'dataset'),
            (importer>>'dataset', lda_trans<<'dataset'),
            (lda_trainer>>'result', lda_trans<<'learned_model'),
            (importer>>'dataset', lda_tester<<'dataset'),
            (lda_trainer>>'result', lda_tester<<'learned_model'),
            (importer>>'dataset', lda_pred<<'dataset'),
            (lda_trainer>>'result', lda_pred<<'learned_model'),
        ])