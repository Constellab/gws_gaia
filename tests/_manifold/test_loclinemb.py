from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      TaskRunner)
from gws_gaia import LocallyLinearEmbeddingTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Locally linear embedding model")
        dataset = DataProvider.get_digits_dataset()

        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'dataset': dataset},
            task_type=LocallyLinearEmbeddingTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        print(trainer_result)
