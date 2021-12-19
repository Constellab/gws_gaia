from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import GaussianMixturePredictor, GaussianMixtureTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian Mixture")
        dataset = GWSGaiaTestHelper.get_dataset(index=6, header=0, targets=[])

        # run trainer
        tester = TaskRunner(
            params={
                'nb_components': 2,
                'covariance_type': 'full'
            },
            inputs={'dataset': dataset},
            task_type=GaussianMixtureTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=GaussianMixturePredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
