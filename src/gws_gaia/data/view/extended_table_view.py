from typing import Union
from pandas import DataFrame

from gws_core import Table, TableView, IntParam, ViewSpecs

class ExtendedTableView(TableView):
    """
    Class table view.

    The view model is:
    ```
    {
        "type": "extended-table",
        "data": dict,
        "targets": dict,
        "from_row": int,
        "number_of_rows_per_page": int,
        "from_column": int,
        "number_of_columns_per_page": int,
        "total_number_of_rows": int,
        "total_number_of_columns": int,
    }
    ```
    """

    _type = "extended-table"
    _data: DataFrame
    _targets: DataFrame

    def check_and_set_data(self, data: "Dataset", *args, **kwargs):
        from ..dataset import Dataset
        if not isinstance(data, Dataset):
            raise BadRequestException("The data must be a Dataset resource")

        self._data = data.get_features()
        self._targets = data.get_targets()

    def _slice_targets(self, from_row_index: int = 0, to_row_index: int = 49) -> dict:
        n = min(self._targets.shape[1], 49)
        return self._targets.iloc[from_row_index:to_row_index, 0:n].to_dict('list')
        
    def to_dict(self, from_row: int = 1, number_of_rows_per_page: int = 50, from_column: int = 1,
                number_of_columns_per_page: int = 50, **kwargs) -> dict:
        total_number_of_rows = self._data.shape[0]
        total_number_of_columns = self._data.shape[1]
        from_row_index = from_row - 1
        from_column_index = from_column - 1
        to_row_index = from_row_index + number_of_rows_per_page - 1
        to_column_index = from_column_index + number_of_columns_per_page - 1

        data = self._slice_data(
            from_row_index=from_row_index,
            to_row_index=to_row_index,
            from_column_index=from_column_index,
            to_column_index=to_column_index
        )

        targets = self._slice_targets(
            from_row_index=from_row_index,
            to_row_index=to_row_index,
        )

        return {
            "type": self._type,
            "data": data,
            "targets": targets,
            "from_row": from_row_index,
            "number_of_rows_per_page": number_of_rows_per_page,
            "from_column": from_column_index,
            "number_of_columns_per_page": number_of_columns_per_page,
            "total_number_of_rows": total_number_of_rows,
            "total_number_of_columns": total_number_of_columns,
        }
