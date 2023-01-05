# design
from .base.helper.training_design_helper import TrainingDesignHelper
# lme
from .gpboost.lme.helper.lme_design_helper import LMEDesignHelper
from .gpboost.lme.lme import LMETrainer, LMETrainerResult
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
# knn
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
