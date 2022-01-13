from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner, ViewTester)
from gws_gaia import AgglomerativeClusteringTrainer
from gws_core.extra import DataProvider
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Agglomerative clustering")
        #dataset = GWSGaiaTestHelper.get_dataset(index=1, header=0, targets=['target1', 'target2'])
        dataset = DataProvider.get_iris_dataset()        
        #--------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            task_type=AgglomerativeClusteringTrainer,
            params={'nb_clusters': 3},
            inputs={'dataset': dataset}
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        #--------------------------------------------------------------------
        # run views
        tester = ViewTester(
            view=trainer_result.view_labels_as_table({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "table-view")

        tester = ViewTester(
            view=trainer_result.view_labels_as_2d_plot({})
        )
        dic = tester.to_dict()
        self.assertEqual(dic["type"], "scatter-plot-2d-view")
        #--------------------------------------------------------------------

        print(trainer_result)
