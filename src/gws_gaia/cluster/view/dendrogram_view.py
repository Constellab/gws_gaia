

from typing import Any, List

import numpy
from pandas import DataFrame
from scipy.cluster import hierarchy

from gws_core import StrParam, ViewSpecs, View, TableView, BadRequestException


class DendrogramView(TableView):
    """
    DendrogramView

    Show a table as a dendrogram

    The view model is:
    ------------------

    ```
    {
        "type": "dendrogram-view",
        "data": dict
    }
    ```
    """

    _type: str = "dendrogram-view"
    _data: "AgglomerativeClusteringResult"
    
    def check_and_set_data(self, data):
        from ..aggclust import AgglomerativeClusteringResult
        if isinstance(data, AgglomerativeClusteringResult):
            raise BadRequestException("The data must be an instance of AgglomerativeClusteringResult")
        self._data = data

    def _compute_linkage():
        """ 
            Create linkage matrix
            from: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        """
        model = self._data
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)
        return linkage_matrix
        # Plot the corresponding dendrogram
        # dendrogram(linkage_matrix, **kwargs)

    def _tree_to_newick(self, node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
        else:
            if len(newick) > 0:
                newick = "):%.2f%s" % (parentdist - node.dist, newick)
            else:
                newick = ");"
            newick = self._tree_to_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = self._tree_to_newick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
            newick = "(%s" % (newick)
            return newick
    
    def _compute_tree(self):
        #Z = linkage(self._data, 'ward')
        Z = self._compute_linkage()
        tree = hierarchy.to_tree(Z,False)
    
    def _compute_newick(self):
        tree = self._compute_tree()
        leaf_names = self._data.get_labels()
        return self._tree_to_newick(tree, "", tree.dist, leaf_names)

    def to_dict(self, *args, **kwargs) -> dict:
        return {
            **super().to_dict(**kwargs),
            "data": {
                "newick": self._compute_newick
            }
        }