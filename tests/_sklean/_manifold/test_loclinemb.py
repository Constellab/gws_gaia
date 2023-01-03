from gws_core import (BaseTestCase, ConfigParams, Table, File, GTest,
                      TaskRunner)
from gws_gaia import LocallyLinearEmbeddingTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Locally linear embedding model")
        table = DataProvider.get_digits_table()

        # run trainer
        tester = TaskRunner(
            params={
                'nb_components': 2,
            },
            inputs={'table': table},
            task_type=LocallyLinearEmbeddingTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        print(trainer_result)
