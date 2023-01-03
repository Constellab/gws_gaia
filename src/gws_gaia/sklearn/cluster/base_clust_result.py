# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import resource_decorator

from ..base.base_unsup import BaseUnsupervisedResult

# *****************************************************************************
#
# BaseClusteringResult
#
# *****************************************************************************


@resource_decorator("BaseClusteringResult", hide=True)
class BaseClusteringResult(BaseUnsupervisedResult):
    """ AgglomerativeClusteringResult """

    CLUSTER_TABLE_NAME = "Cluster table"

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_cluster_table()

    def _create_cluster_table(self):
        mdl = self.get_result()
        table = self.get_training_set()

        row_tags = table.get_row_tags()
        for i, tag in enumerate(row_tags):
            tag["label_"] = mdl.labels_[i]

        table.name = self.CLUSTER_TABLE_NAME
        table.set_all_row_tags(row_tags)
        self.add_resource(table)
