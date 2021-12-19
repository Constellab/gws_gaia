from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import KNNRegressorPredictor, KNNRegressorTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("K-nearest neighbors regressor")
        dataset = GWSGaiaTestHelper.get_dataset(index=2, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={'nb_neighbors': 5},
            inputs={'dataset': dataset},
            task_type=KNNRegressorTrainer
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
            task_type=KNNRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
