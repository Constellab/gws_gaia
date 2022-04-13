
# cluster
from .cluster.aggclust import AgglomerativeClusteringResult, AgglomerativeClusteringTrainer
from .cluster.kmeans import KMeansResult, KMeansTrainer, KMeansPredictor
# discrinimant analysis
from .da.linearda import LDAResult, LDATrainer, LDATransformer, LDAPredictor
from .da.quadrada import QDAResult, QDATrainer, QDAPredictor
# decomposition
from .decomp.ica import ICAResult, ICATrainer
from .decomp.pca import PCATrainerResult, PCATrainer, PCATransformer
from .decomp.pcoa import PCoATrainerResult, PCoATrainer
from .decomp.pls import PLSTrainerResult, PLSTrainer, PLSPredictor, PLSTransformer
# ensemble
from .ensemble.adaboostclass import (AdaBoostClassifierPredictor, AdaBoostClassifierResult,
                                        AdaBoostClassifierTrainer)
from .ensemble.adaboostreg import (AdaBoostRegressorResult, AdaBoostRegressorTrainer,
                                        AdaBoostRegressorPredictor )
from .ensemble.extratreeclass import (ExtraTreesClassifierResult, ExtraTreesClassifierTrainer,
                                        ExtraTreesClassifierPredictor)
from .ensemble.extratreereg import (ExtraTreesRegressorResult, ExtraTreesRegressorTrainer,
                                        ExtraTreesRegressorPredictor)
from .ensemble.gradboostclass import (GradientBoostingClassifierResult, GradientBoostingClassifierTrainer,
                                        GradientBoostingClassifierPredictor)
from .ensemble.gradboostreg import (GradientBoostingRegressorResult, GradientBoostingRegressorTrainer,
                                        GradientBoostingRegressorPredictor)
from .ensemble.randforestclass import (RandomForestClassifierResult, RandomForestClassifierTrainer,
                                        RandomForestClassifierPredictor)
from .ensemble.randforestreg import (RandomForestRegressorResult, RandomForestRegressorTrainer,
                                        RandomForestRegressorPredictor)
# gaussian process
from .gaussian_process.gaussprocclass import (GaussianProcessClassifierResult, GaussianProcessClassifierTrainer,
                                                GaussianProcessClassifierPredictor)
from .gaussian_process.gaussprocreg import (GaussianProcessRegressorResult, GaussianProcessRegressorTrainer,
                                                GaussianProcessRegressorPredictor)
#kernel ridge
from .kernel_ridge.kernridge import (KernelRidgeResult, KernelRidgeTrainer, KernelRidgePredictor)
#kNN
from .knn.kneighclass import (KNNClassifierResult, KNNClassifierTrainer, KNNClassifierPredictor)
from .knn.kneighreg import (KNNRegressorResult, KNNRegressorTrainer, KNNRegressorPredictor)
#lm
from .lm.elastnet import (ElasticNetResult, ElasticNetTrainer, ElasticNetPredictor)
from .lm.lasso import (LassoResult, LassoTrainer, LassoPredictor)
from .lm.linearreg import (LinearRegressionResult, LinearRegressionTrainer, LinearRegressionPredictor)
from .lm.logreg import (LogisticRegressionResult, LogisticRegressionTrainer, LogisticRegressionPredictor)
from .lm.ridgeclass import (RidgeClassifierResult, RidgeClassifierTrainer, RidgeClassifierPredictor)
from .lm.ridgereg import (RidgeRegressionResult, RidgeRegressionTrainer, RidgeRegressionPredictor)
from .lm.sgdclass import (SGDClassifierResult, SGDClassifierTrainer, SGDClassifierPredictor)
from .lm.sgdreg import (SGDRegressorResult, SGDRegressorTrainer, SGDRegressorPredictor)
# manifold
from .manifold.loclinemb import LocallyLinearEmbeddingResult, LocallyLinearEmbeddingTrainer
# mixture
from .mixture.gaussmixt import (GaussianMixtureResult, GaussianMixtureTrainer, GaussianMixturePredictor)
# naive bayes
from .naive_bayes.bernoulnb import (BernoulliNaiveBayesClassifierResult, BernoulliNaiveBayesClassifierTrainer,
                                        BernoulliNaiveBayesClassifierPredictor)
from .naive_bayes.gaussiannb import (GaussianNaiveBayesResult, GaussianNaiveBayesTrainer,
                                        GaussianNaiveBayesPredictor)
from .naive_bayes.multinomnb import (MultinomialNaiveBayesClassifierResult, MultinomialNaiveBayesClassifierTrainer,
                                        MultinomialNaiveBayesClassifierPredictor)
# svm
from .svm.svc import (SVCResult, SVCTrainer, SVCPredictor)
from .svm.svr import (SVRResult, SVRTrainer, SVRPredictor)
# tree
from .tree.decistreeclass import (DecisionTreeClassifierResult, DecisionTreeClassifierTrainer,
                                    DecisionTreeClassifierPredictor)
from .tree.decistreereg import (DecisionTreeRegressorResult, DecisionTreeRegressorTrainer,
                                    DecisionTreeRegressorPredictor)
# tf
from .tensorflow.averagepoollayers import (AveragePooling1D as TFAveragePooling1D,
                                    AveragePooling2D as TFAveragePooling2D,
                                    AveragePooling3D as TFAveragePooling3D )
from .tensorflow.convlayers import (Conv1D as TFConv1D,
                            Conv2D as TFConv2D,
                            Conv3D as TFConv3D)
from .tensorflow.corelayers import (Dense as TFDense,
                            Activation as TFActivation,
                            Masking as TFMasking,
                            Embedding as TFEmbedding,
                            Dropout as TFDropout,
                            Flatten as TFFlatten)
from .tensorflow.data import (Tensor as TFTensor,
                        DeepModel as TFDeepModel,
                        InputConverter as TFInputConverter)
from .tensorflow.deepmodeler import (DeepModelBuilder as TFDeepModelBuilder,
                            DeepModelCompiler as TFDeepModelCompiler,
                            DeepModelerTrainer as TFDeepModelerTrainer,
                            DeepModelerPredictor as TFDeepModelerPredictor)
from .tensorflow.globalaveragepoollayers import (GlobalAveragePooling1D as TFGlobalAveragePooling1D,
                                            GlobalAveragePooling2D as TFGlobalAveragePooling2D,
                                            GlobalAveragePooling3D as TFGlobalAveragePooling3D)
from .tensorflow.globalmaxpoollayers import (GlobalMaxPooling1D as TFGlobalMaxPooling1D,
                                        GlobalMaxPooling2D as TFGlobalMaxPooling2D,
                                        GlobalMaxPooling3D as TFGlobalMaxPooling3D)
from .tensorflow.importingpkl import (PickleImporter as TFPickleImporter,
                                Preprocessor as TFPreprocessor,
                                AdhocExtractor as TFAdhocExtractor)
from .tensorflow.maxpoollayers import (MaxPooling1D as TFMaxPooling1D,
                                MaxPooling2D as TFMaxPooling2D,
                                MaxPooling3D as TFMaxPooling3D)
from .tensorflow.recurrentlayers import (LSTM as TFLSTM, GRU as TFGRU, SimpleRNN as TFSimpleRNN)