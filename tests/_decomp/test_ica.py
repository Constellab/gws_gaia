from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner)
from gws_gaia import ICATrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Independant Component Analysis (ICA)")
        dataset = DataProvider.get_digits_dataset()

        # --------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 7},
            inputs={'dataset': dataset},
            task_type=ICATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        print(trainer_result)
