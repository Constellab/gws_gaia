from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TaskRunner)
from gws_gaia import GaussianMixtureTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian Mixture")
        table = GWSGaiaTestHelper.get_table(index=6, header=0, targets=[])

        # run trainer
        tester = TaskRunner(
            params={
                'nb_components': 2,
                'covariance_type': 'full'
            },
            inputs={'table': table},
            task_type=GaussianMixtureTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        print(trainer_result)
