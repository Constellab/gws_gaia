
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import LocallyLinearEmbeddingTrainer
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

        
    async def test_process(self):
        GTest.print("Locally linear embedding model")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        p0 = DatasetLoader()
        p1 = LocallyLinearEmbeddingTrainer()

        proto = Protocol(
            processes = {
                'p0' : p0,
                'p1' : p1
            },
            connectors = [
                p0>>'dataset' | p1<<'dataset',
            ]
        )
        
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', [])
        p0.set_param("file_path", os.path.join(test_dir, "./digits.csv"))
        p1.set_param('nb_components', 2)

        experiment: Experiment = Experiment(
            protocol=proto, study=GTest.study, user=GTest.user)
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)                  

        r = p1.output['result']
        print(r)