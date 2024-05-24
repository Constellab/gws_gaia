import numpy
from gws_core import BaseTestCaseLight, TaskRunner
from gws_gaia import PCoATrainer
from gws_gaia.data_provider.data_provider import DataProvider


class TestTrainer(BaseTestCaseLight):

    def test_pcoa(self):
        self.print("Principal Coordinate Analysis (PCoA)")
        distance_table = DataProvider.get_distance_table()

        # ---------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'nb_components': 2},
            inputs={'distance_table': distance_table},
            task_type=PCoATrainer
        )
        outputs = tester.run()
        trainer_result = outputs['result']

        table = trainer_result.get_variance_table()

        self.assertTrue(numpy.all(numpy.isclose(
            table.get_data().to_numpy(), [[0.087764], [0.060941]], atol=1e-3)))

        table = trainer_result.get_transformed_table()
