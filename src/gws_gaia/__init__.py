from .base.helper.training_design_helper import TrainingDesignHelper
# pcoa
from .skbio.pcoa import PCoATrainer, PCoATrainerResult
# cluster
from .sklearn.cluster.aggclust import (AgglomerativeClusteringResult,
                                       AgglomerativeClusteringTrainer)
from .sklearn.cluster.kmeans import (KMeansPredictor, KMeansResult,
                                     KMeansTrainer)
# discrinimant analysis
from .sklearn.da.linearda import LDAPredictor, LDATrainer, LDATrainerResult
from .sklearn.da.plsda import PLSDAPredictor, PLSDATrainer, PLSDATrainerResult
# decomposition
from .sklearn.decomp.ica import ICAResult, ICATrainer
from .sklearn.decomp.pca import PCATrainer, PCATrainerResult
from .sklearn.decomp.plsr import PLSPredictor, PLSTrainer, PLSTrainerResult
# ensemble
from .sklearn.ensemble.adaboostclass import (AdaBoostClassifierPredictor,
                                             AdaBoostClassifierResult,
                                             AdaBoostClassifierTrainer)
from .sklearn.ensemble.adaboostreg import (AdaBoostRegressorPredictor,
                                           AdaBoostRegressorResult,
                                           AdaBoostRegressorTrainer)
from .sklearn.ensemble.extratreeclass import (ExtraTreesClassifierPredictor,
                                              ExtraTreesClassifierResult,
                                              ExtraTreesClassifierTrainer)
from .sklearn.ensemble.extratreereg import (ExtraTreesRegressorPredictor,
                                            ExtraTreesRegressorResult,
                                            ExtraTreesRegressorTrainer)
from .sklearn.ensemble.gradboostclass import (
    GradientBoostingClassifierPredictor, GradientBoostingClassifierResult,
    GradientBoostingClassifierTrainer)
from .sklearn.ensemble.gradboostreg import (GradientBoostingRegressorPredictor,
                                            GradientBoostingRegressorResult,
                                            GradientBoostingRegressorTrainer)
from .sklearn.ensemble.randforestclass import (RandomForestClassifierPredictor,
                                               RandomForestClassifierResult,
                                               RandomForestClassifierTrainer)
from .sklearn.ensemble.randforestreg import (RandomForestRegressorPredictor,
                                             RandomForestRegressorResult,
                                             RandomForestRegressorTrainer)
# gaussian process
from .sklearn.gaussian_process.gaussprocclass import (
    GaussianProcessClassifierPredictor, GaussianProcessClassifierResult,
    GaussianProcessClassifierTrainer)
from .sklearn.gaussian_process.gaussprocreg import (
    GaussianProcessRegressorPredictor, GaussianProcessRegressorResult,
    GaussianProcessRegressorTrainer)
# kernel ridge
from .sklearn.kernel_ridge.kernridge import (KernelRidgePredictor,
                                             KernelRidgeResult,
                                             KernelRidgeTrainer)
# kNN
from .sklearn.knn.kneighclass import (KNNClassifierPredictor,
                                      KNNClassifierResult,
                                      KNNClassifierTrainer)
from .sklearn.knn.kneighreg import (KNNRegressorPredictor, KNNRegressorResult,
                                    KNNRegressorTrainer)
# lm
from .sklearn.lm.elastnet import (ElasticNetPredictor, ElasticNetResult,
                                  ElasticNetTrainer)
from .sklearn.lm.lasso import LassoPredictor, LassoResult, LassoTrainer
from .sklearn.lm.linearreg import (LinearRegressionPredictor,
                                   LinearRegressionResult,
                                   LinearRegressionTrainer)
from .sklearn.lm.logreg import (LogisticRegressionPredictor,
                                LogisticRegressionResult,
                                LogisticRegressionTrainer)
from .sklearn.lm.ridgeclass import (RidgeClassifierPredictor,
                                    RidgeClassifierResult,
                                    RidgeClassifierTrainer)
from .sklearn.lm.ridgereg import (RidgeRegressionPredictor,
                                  RidgeRegressionResult,
                                  RidgeRegressionTrainer)
from .sklearn.lm.sgdclass import (SGDClassifierPredictor, SGDClassifierResult,
                                  SGDClassifierTrainer)
from .sklearn.lm.sgdreg import (SGDRegressorPredictor, SGDRegressorResult,
                                SGDRegressorTrainer)
# manifold
from .sklearn.manifold.loclinemb import (LocallyLinearEmbeddingResult,
                                         LocallyLinearEmbeddingTrainer)
# mixture
from .sklearn.mixture.gaussmixt import (GaussianMixtureResult,
                                        GaussianMixtureTrainer)
# naive bayes
from .sklearn.naive_bayes.bernoulnb import (
    BernoulliNaiveBayesClassifierPredictor,
    BernoulliNaiveBayesClassifierResult, BernoulliNaiveBayesClassifierTrainer)
from .sklearn.naive_bayes.gaussiannb import (GaussianNaiveBayesPredictor,
                                             GaussianNaiveBayesResult,
                                             GaussianNaiveBayesTrainer)
from .sklearn.naive_bayes.multinomnb import (
    MultinomialNaiveBayesClassifierPredictor,
    MultinomialNaiveBayesClassifierResult,
    MultinomialNaiveBayesClassifierTrainer)
# svm
from .sklearn.svm.svc import SVCPredictor, SVCResult, SVCTrainer
from .sklearn.svm.svr import SVRPredictor, SVRResult, SVRTrainer
# tree
from .sklearn.tree.decistreeclass import (DecisionTreeClassifierPredictor,
                                          DecisionTreeClassifierResult,
                                          DecisionTreeClassifierTrainer)
from .sklearn.tree.decistreereg import (DecisionTreeRegressorPredictor,
                                        DecisionTreeRegressorResult,
                                        DecisionTreeRegressorTrainer)

# # tf
# from .tensorflow.averagepoollayers import \
#     AveragePooling1D as TFAveragePooling1D
# from .tensorflow.averagepoollayers import \
#     AveragePooling2D as TFAveragePooling2D
# from .tensorflow.averagepoollayers import \
#     AveragePooling3D as TFAveragePooling3D
# from .tensorflow.convlayers import Conv1D as TFConv1D
# from .tensorflow.convlayers import Conv2D as TFConv2D
# from .tensorflow.convlayers import Conv3D as TFConv3D
# from .tensorflow.corelayers import Activation as TFActivation
# from .tensorflow.corelayers import Dense as TFDense
# from .tensorflow.corelayers import Dropout as TFDropout
# from .tensorflow.corelayers import Embedding as TFEmbedding
# from .tensorflow.corelayers import Flatten as TFFlatten
# from .tensorflow.corelayers import Masking as TFMasking
# from .tensorflow.data import DeepModel as TFDeepModel
# from .tensorflow.data import InputConverter as TFInputConverter
# from .tensorflow.data import Tensor as TFTensor
# from .tensorflow.deepmodeler import DeepModelBuilder as TFDeepModelBuilder
# from .tensorflow.deepmodeler import DeepModelCompiler as TFDeepModelCompiler
# from .tensorflow.deepmodeler import \
#     DeepModelerPredictor as TFDeepModelerPredictor
# from .tensorflow.deepmodeler import DeepModelerTrainer as TFDeepModelerTrainer
# from .tensorflow.globalaveragepoollayers import \
#     GlobalAveragePooling1D as TFGlobalAveragePooling1D
# from .tensorflow.globalaveragepoollayers import \
#     GlobalAveragePooling2D as TFGlobalAveragePooling2D
# from .tensorflow.globalaveragepoollayers import \
#     GlobalAveragePooling3D as TFGlobalAveragePooling3D
# from .tensorflow.globalmaxpoollayers import \
#     GlobalMaxPooling1D as TFGlobalMaxPooling1D
# from .tensorflow.globalmaxpoollayers import \
#     GlobalMaxPooling2D as TFGlobalMaxPooling2D
# from .tensorflow.globalmaxpoollayers import \
#     GlobalMaxPooling3D as TFGlobalMaxPooling3D
# from .tensorflow.importingpkl import AdhocExtractor as TFAdhocExtractor
# from .tensorflow.importingpkl import PickleImporter as TFPickleImporter
# from .tensorflow.importingpkl import Preprocessor as TFPreprocessor
# from .tensorflow.maxpoollayers import MaxPooling1D as TFMaxPooling1D
# from .tensorflow.maxpoollayers import MaxPooling2D as TFMaxPooling2D
# from .tensorflow.maxpoollayers import MaxPooling3D as TFMaxPooling3D
# from .tensorflow.recurrentlayers import GRU as TFGRU
# from .tensorflow.recurrentlayers import LSTM as TFLSTM
# from .tensorflow.recurrentlayers import SimpleRNN as TFSimpleRNN
