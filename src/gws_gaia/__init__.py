
# cluster
from .cluster.aggclust import AgglomerativeClusteringResult, AgglomerativeClusteringTrainer, AgglomerativeClustering
from .cluster.kmeans import KMeansResult, KMeansTrainer, KMeansPredictor
# discrinimant analysis
from .da.linearda import LDAResult, LDATrainer, LDATransformer, LDATester, LDAPredictor
from .da.quadrada import QDAResult, QDATrainer, QDATester, QDAPredictor
# dataset

from .data.core import Tuple
from .data.dataset import Dataset, DatasetExporter, DatasetImporter, DatasetLoader, DatasetDumper
# decomposition
from .decomp.ica import ICAResult, ICATrainer
from .decomp.pca import PCAResult, PCATrainer, PCATransformer
from .decomp.pls import PLSResult, PLSTrainer, PLSPredictor, PLSTester
# ensemble
from .ensemble.adaboostclass import (AdaBoostClassifierPredictor, AdaBoostClassifierResult, 
                                        AdaBoostClassifierTester, AdaBoostClassifierTrainer)
from .ensemble.adaboostreg import (AdaBoostRegressorResult, AdaBoostRegressorTrainer,
                                        AdaBoostRegressorPredictor, AdaBoostRegressorTester )
from .ensemble.extratreeclass import (ExtraTreesClassifierResult, ExtraTreesClassifierTrainer, 
                                        ExtraTreesClassifierPredictor, ExtraTreesClassifierTester)
from .ensemble.extratreereg import (ExtraTreesRegressorResult, ExtraTreesRegressorTrainer, 
                                        ExtraTreesRegressorPredictor, ExtraTreesRegressorTester)
from .ensemble.gradboostclass import (GradientBoostingClassifierResult, GradientBoostingClassifierTrainer, 
                                        GradientBoostingClassifierPredictor, GradientBoostingClassifierTester)
from .ensemble.gradboostreg import (GradientBoostingRegressorResult, GradientBoostingRegressorTrainer, 
                                        GradientBoostingRegressorPredictor, GradientBoostingRegressorTester)
from .ensemble.randforestclass import (RandomForestClassifierResult, RandomForestClassifierTrainer, 
                                        RandomForestClassifierPredictor, RandomForestClassifierTester)    
from .ensemble.randforestreg import (RandomForestRegressorResult, RandomForestRegressorTrainer, 
                                        RandomForestRegressorPredictor, RandomForestRegressorTester)         
# gaussian process
from .gaussian_process.gaussprocclass import (GaussianProcessClassifierResult, GaussianProcessClassifierTrainer, 
                                                GaussianProcessClassifierPredictor, GaussianProcessClassifierTester) 
from .gaussian_process.gaussprocreg import (GaussianProcessRegressorResult, GaussianProcessRegressorTrainer, 
                                                GaussianProcessRegressorPredictor, GaussianProcessRegressorTester) 
#kernel ridge
from .kernel_ridge.kernridge import (KernelRidgeResult, KernelRidgeTrainer, KernelRidgePredictor, KernelRidgeTester) 
#kNN
from .knn.kneighclass import (KNNClassifierResult, KNNClassifierTrainer, KNNClassifierPredictor, KNNClassifierTester) 
from .knn.kneighreg import (KNNRegressorResult, KNNRegressorTrainer, KNNRegressorPredictor, KNNRegressorTester) 
#lm
from .lm.elastnet import (ElasticNetResult, ElasticNetTrainer, ElasticNetPredictor, ElasticNetTester)
from .lm.lasso import (LassoResult, LassoTrainer, LassoPredictor, LassoTester)
from .lm.linearreg import (LinearRegressionResult, LinearRegressionTrainer, LinearRegressionPredictor, LinearRegressionTester)
from .lm.logreg import (LogisticRegressionResult, LogisticRegressionTrainer, LogisticRegressionPredictor, LogisticRegressionTester)
from .lm.ridgeclass import (RidgeClassifierResult, RidgeClassifierTrainer, RidgeClassifierPredictor, RidgeClassifierTester)
from .lm.ridgereg import (RidgeRegressionResult, RidgeRegressionTrainer, RidgeRegressionPredictor, RidgeRegressionTester)
from .lm.sgdclass import (SGDClassifierResult, SGDClassifierTrainer, SGDClassifierPredictor, SGDClassifierTester)
from .lm.sgdreg import (SGDRegressorResult, SGDRegressorTrainer, SGDRegressorPredictor, SGDRegressorTester)
# manifold
from .manifold.loclinemb import LocallyLinearEmbeddingResult, LocallyLinearEmbeddingTrainer
# mixture
from .mixture.gaussmixt import (GaussianMixtureResult, GaussianMixtureTrainer, GaussianMixturePredictor)
# naive bayes
from .naive_bayes.bernoulnb import (BernoulliNaiveBayesClassifierResult, BernoulliNaiveBayesClassifierTrainer, 
                                        BernoulliNaiveBayesClassifierPredictor, BernoulliNaiveBayesClassifierTester)
from .naive_bayes.gaussiannb import (GaussianNaiveBayesResult, GaussianNaiveBayesTrainer, 
                                        GaussianNaiveBayesPredictor, GaussianNaiveBayesTester)
from .naive_bayes.multinomnb import (MultinomialNaiveBayesClassifierResult, MultinomialNaiveBayesClassifierTrainer, 
                                        MultinomialNaiveBayesClassifierPredictor, MultinomialNaiveBayesClassifierTester)
# svm
from .svm.svc import (SVCResult, SVCTrainer, SVCPredictor, SVCTester)
from .svm.svr import (SVRResult, SVRTrainer, SVRPredictor, SVRTester)
# tree
from .tree.decistreeclass import (DecisionTreeClassifierResult, DecisionTreeClassifierTrainer, 
                                    DecisionTreeClassifierPredictor, DecisionTreeClassifierTester)
from .tree.decistreereg import (DecisionTreeRegressorResult, DecisionTreeRegressorTrainer, 
                                    DecisionTreeRegressorPredictor, DecisionTreeRegressorTester)
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
from .tensorflow.deepmodeler import (DeepModelerBuilder as TFDeepModelerBuilder,
                            DeepModelerCompiler as TFDeepModelerCompiler,
                            DeepModelerTrainer as TFDeepModelerTrainer,
                            DeepModelerTester as TFDeepModelerTester,
                            DeepModelerPredictor as TFDeepModelerPredictor)
from .tensorflow.globalaveragepoollayers import (GlobalAveragePooling1D as TFGlobalAveragePooling1D,
                                            GlobalAveragePooling2D as TFGlobalAveragePooling2D,
                                            GlobalAveragePooling3D as TFGlobalAveragePooling3D)
from .tensorflow.globalmaxpoollayers import (GlobalMaxPooling1D as TFGlobalMaxPooling1D,
                                        GlobalMaxPooling2D as TFGlobalMaxPooling2D,
                                        GlobalMaxPooling3D as TFGlobalMaxPooling3D)
from .tensorflow.importingpkl import (ImporterPKL as TFImporterPKL,
                                Preprocessor as TFPreprocessor,
                                AdhocExtractor as TFAdhocExtractor)
from .tensorflow.maxpoollayers import (MaxPooling1D as TFMaxPooling1D,
                                MaxPooling2D as TFMaxPooling2D,
                                MaxPooling3D as TFMaxPooling3D)
from .tensorflow.recurrentlayers import (LSTM as TFLSTM, GRU as TFGRU, SimpleRNN as TFSimpleRNN)