# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws.protocol import Protocol
from gws.study import Study
from gws.user import User

from gaia.data import Tuple
from gaia.dataset import Dataset, Importer
from gaia.linearda import   Trainer as LDATrainer, \
                            Predictor as LDAPredictor, \
                            Tester as LDATester, \
                            Transformer as LDATransformer
from gaia.pca import Trainer as PCATrainer, Transformer as PCATransformer

class LDAProto(Protocol):
    def __init__(self, *args, user = None, **kwargs): 
        super().__init__(*args, user=user, **kwargs)
        if not self.is_built:
            importer = Importer()
            lda_trainer = LDATrainer()
            lda_pred = LDAPredictor()
            lda_tester = LDATester()
            lda_trans = LDATransformer()
            pca_trainer = PCATrainer()
            pca_trans = PCATransformer()

            processes = {
                'importer' : importer,
                'lda_trainer' : lda_trainer,
                'lda_predictor' : lda_pred,
                'lda_tester' : lda_tester,
                'lda_transf' : lda_trans,
                'pca_trainer' : pca_trainer,
                'pca_transf' : pca_trans
            }
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

            self.set_title("LDA protocol")
            self.set_description("This is a short LDA protocol provided to you to see how linear discrimant analysis could be implemented.")

            self._build(
                processes = processes,
                connectors = connectors,
                interfaces = {},
                outerfaces = {},
                user = user,
                **kwargs
            )

def lda_pca_experiment(data_file, delimiter=",", header=0, target=[], ncomp=2):
    
    proto = LDAProto()
    importer = proto.processes["importer"]
    lda_trainer = proto.processes["lda_trainer"]
    pca_trainer = proto.processes["pca_trainer"]

    importer.set_param("delimiter", delimiter)
    importer.set_param("header", header)
    importer.set_param('targets', target) 
    importer.set_param("file_path", data_file)
    lda_trainer.set_param('nb_components', ncomp)
    pca_trainer.set_param('nb_components', ncomp)
    proto.save()
    
    e = proto.create_experiment(study=Study.get_default_instance(), user=User.get_sysuser())
    e.set_title("Short LDA and PCA experiment")
    e.set_description("This is a short LDA protocol provided to you to see how linear discrimant analysis could be implemented.")
    e.save()
    
    return e