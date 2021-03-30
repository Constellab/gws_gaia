# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource, Protocol, Study, User

from gaia.data import Tuple
from gaia.dataset import Dataset, Importer
from gaia.linearda import   Trainer as LDATrainer, \
                            Predictor as LDAPredictor, \
                            Tester as LDATester, \
                            Transformer as LDATransformer
from gaia.pca import Trainer as PCATrainer, Transformer as PCATransformer

def lda_pca_experiment(data_file, delimiter=",", header=0, target=[], ncomp=2):
    importer = Importer()
    lda_trainer = LDATrainer()
    lda_pred = LDAPredictor()
    lda_tester = LDATester()
    lda_trans = LDATransformer()
    pca_trainer = PCATrainer()
    pca_trans = PCATransformer()

    proto = Protocol(
        processes = {
            'importer' : importer,
            'lda_trainer' : lda_trainer,
            'lda_predictor' : lda_pred,
            'lda_tester' : lda_tester,
            'lda_transf' : lda_trans,
            'pca_trainer' : pca_trainer,
            'pca_transf' : pca_trans
        },
        connectors = [
            importer>>'dataset'   | pca_trainer<<'dataset',                
            importer>>'dataset'   | pca_trans<<'dataset',                
            pca_trainer>>'result' | pca_trans<<'learned_model',                
            importer>>'dataset'   | lda_trainer<<'dataset',
            importer>>'dataset'   | lda_trans<<'dataset',
            lda_trainer>>'result' | lda_trans<<'learned_model',
            importer>>'dataset'   | lda_tester<<'dataset',
            lda_trainer>>'result' | lda_tester<<'learned_model',
            importer>>'dataset'   | lda_pred<<'dataset',
            lda_trainer>>'result' | lda_pred<<'learned_model',
        ]
    )

    importer.set_param("delimiter", delimiter)
    importer.set_param("header", header)
    importer.set_param('targets', target) 
    importer.set_param("file_path", data_file)
    lda_trainer.set_param('nb_components', ncomp)
    pca_trainer.set_param('nb_components', ncomp)

    def _end(*args, **kwargs):
        pass
    
    proto.set_title("LDA protocol")
    proto.set_description("This is a short LDA protocol provided to you to see how linear discrimant analysis could be implemented.")
    proto.on_end(_end)
    proto.save()
    
    e = proto.create_experiment(study=Study.get_default_instance(), user=User.get_sysuser())
    e.set_title("Short LDA and PCA experiment")
    e.set_description("This is a short LDA protocol provided to you to see how linear discrimant analysis could be implemented.")
    e.save()
    
    return e