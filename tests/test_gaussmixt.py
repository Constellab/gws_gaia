import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import GaussianMixtureTrainer, GaussianMixturePredictor
from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        GTest.print("Gaussian mixture model")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = DatasetLoader()
        p2 = GaussianMixtureTrainer()
        p3 = GaussianMixturePredictor()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2,
                'p3' : p3
            },
            connectors = [
                p0>>'dataset' | p2<<'dataset',
                p1>>'dataset' | p3<<'dataset',
                p2>>'result' | p3<<'learned_model'
            ]
        )

        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        #p0.set_param('targets', [])
        p1.set_param("delimiter", ",")
        p1.set_param("header", 0)
        #p1.set_param('targets', [])
 
        p0.set_param("file_path", os.path.join(test_dir, "./dataset6.csv"))
        p1.set_param("file_path", os.path.join(test_dir, "./dataset6.csv"))
        p2.set_param('nb_components', 2)
        p2.set_param('covariance_type', 'full')

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)                         

        r1 = p2.output['result']
        r2 = p3.output['result']

        print(r1)
        print(r2)