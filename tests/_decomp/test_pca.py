import numpy
from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import PCATrainer, PCATransformer


class TestTrainer(BaseTestCase):

    async def test_pca(self):
        self.print("Principal Component Analysis (PCA)")
        dataset = DataProvider.get_iris_dataset()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'dataset': dataset},
            task_type=PCATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        table = trainer_result.get_variance_table()
        self.assertTrue(numpy.all(numpy.isclose(
            table.get_data().to_numpy(), [[0.9246187232017271], [0.053066483117067804]], atol=1e-3)))

        # --------------------------------------------------------------------
        # run transformer
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=PCATransformer
        )
        outputs = await tester.run()
        transformer_result = outputs['result']

        print(trainer_result)
        print(transformer_result)
