import numpy
from gws_core import (BaseTestCase, ConfigParams, File, GTest, Settings, Table,
                      TableImporter, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import PCATrainer


class TestTrainer(BaseTestCase):

    async def test_pca(self):
        self.print("Principal Component Analysis (PCA)")
        table = DataProvider.get_iris_table(keep_variety=False)

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'table': table},
            task_type=PCATrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        var_table = trainer_result.get_variance_table()
        self.assertTrue(numpy.all(numpy.isclose(
            var_table.get_data().to_numpy(), [[0.9246187232017271], [0.053066483117067804]], atol=1e-3)))