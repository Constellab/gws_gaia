
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import PCATrainer, PCATransformer
from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        GTest.print("Principal Component Analysis (PCA)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = PCATrainer()
        p2 = PCATransformer()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1,
                'p2' : p2
            },
            connectors = [
                p0>>'dataset' | p1<<'dataset',
                p0>>'dataset' | p2<<'dataset',
                p1>>'result' | p2<<'learned_model',
            ]
        )
        
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p1.set_param('nb_components', 2)

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)               

        r1 = p1.output['result']
        r2 = p2.output['result']

        #print(r2.tuple)