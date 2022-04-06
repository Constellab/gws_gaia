# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os

from gws_core import (ConfigParams, Dataset, DatasetImporter, Experiment, File,
                      ProcessSpec, Protocol, Settings, Table, User,
                      protocol_decorator)
from gws_gaia import (LDAPredictor, LDATrainer, LDATransformer, PCATrainer,
                      PCATransformer)


@ protocol_decorator("LDATutorialProto",
                     human_name="LDA Proto",
                     short_description="Tutorial: short LDA and PCA protocol", hide=True)
class LDATutorialProto(Protocol):

    def configure_protocol(self, config_params: ConfigParams) -> None:
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        importer: ProcessSpec = self.add_process(DatasetImporter, 'importer')
        lda_trainer: ProcessSpec = self.add_process(LDATrainer, 'lda_trainer')
        lda_pred: ProcessSpec = self.add_process(LDAPredictor, 'lda_pred')
        lda_trans: ProcessSpec = self.add_process(LDATransformer, 'lda_trans')
        pca_trainer: ProcessSpec = self.add_process(PCATrainer, 'pca_trainer')
        pca_trans: ProcessSpec = self.add_process(PCATransformer, 'pca_trans')

        importer.set_param("delimiter", ",")
        importer.set_param("header", 0)
        importer.set_param("metadata_columns", [{
            "column": "variety",
            "type": Table.CATEGORICAL_TAG_TYPE,
            "keep_in_data": True,
            "is_target": True
        }])

        lda_trainer.set_param('nb_components', 2)
        pca_trainer.set_param('nb_components', 2)

        self.add_connectors([
            (importer >> 'target', pca_trainer << 'dataset'),
            (importer >> 'target', pca_trans << 'dataset'),
            (pca_trainer >> 'result', pca_trans << 'learned_model'),
            (importer >> 'target', lda_trainer << 'dataset'),
            (importer >> 'target', lda_trans << 'dataset'),
            (lda_trainer >> 'result', lda_trans << 'learned_model'),
            (importer >> 'target', lda_pred << 'dataset'),
            (lda_trainer >> 'result', lda_pred << 'learned_model'),
        ])

        self.add_interface('file', importer, 'source')
