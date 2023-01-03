

import os

from gws_core import (Dataset, DatasetImporter, File, Settings, Table,
                      TableImporter)


class DataProvider():

    @classmethod
    def _get_test_data_dir(cls) -> str:
        return Settings.retrieve().get_variable("gws_gaia:testdata_dir")

    @classmethod
    def get_test_data_path(cls, path: str) -> str:
        return os.path.join(cls._get_test_data_dir(), path)

    @classmethod
    def get_distance_table_file(cls) -> File:
        return File(cls.get_test_data_path('distance_matrix.csv'))

    @classmethod
    def get_digits_file(cls) -> File:
        return File(cls.get_test_data_path('digits.csv'))

    @classmethod
    def get_diabetes_file(cls) -> File:
        return File(cls.get_test_data_path('diabetes.csv'))

    @classmethod
    def get_digits_dataset(cls, header=0, targets=[]) -> Dataset:
        return DatasetImporter.call(cls.get_digits_file(), {
            "delimiter": ",",
            "header": header,
            # "targets": targets,
            "metadata_columns": [{
                "column": k,
                "keep_in_data": True,
                "is_target": True
            } for k in targets]
        })

    @classmethod
    def get_diabetes_dataset(cls, header=0, targets=['target']) -> Dataset:
        return DatasetImporter.call(cls.get_diabetes_file(), {
            "delimiter": ",",
            "header": header,
            # "targets": targets,
            "metadata_columns": [{
                "column": k,
                "keep_in_data": True,
                "is_target": True
            } for k in targets]
        })

    @classmethod
    def get_distance_table(cls) -> Table:
        return TableImporter.call(cls.get_distance_table_file(), {
            "delimiter": "tab",
            "header": 0,
            "index_column": 0,
            "metadata_columns": []
        })
