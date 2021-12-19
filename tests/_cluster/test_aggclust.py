from gws_core import (BaseTestCase, ConfigParams, Dataset, ExperimentService,
                      File, IntParam, Settings, TaskRunner)
from gws_gaia import AgglomerativeClusteringTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Agglomerative clustering")
        dataset = GWSGaiaTestHelper.get_dataset(index=1, header=0, targets=['target1', 'target2'])
        tester = TaskRunner(
            task_type=AgglomerativeClusteringTrainer,
            params={'nb_clusters': 2},
            inputs={'dataset': dataset}
        )
        outputs = await tester.run()
        r1 = outputs['result']

        print(r1)
