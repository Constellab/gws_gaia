

import os

from gws_core import Table, TableImporter, File, Settings, Table


class GWSGaiaTestHelper():

    @classmethod
    def get_test_data_dir(cls) -> str:
        return Settings.retrieve().get_variable("gws_gaia:testdata_dir")

    @classmethod
    def get_test_data_path(cls, *path: str) -> str:
        return os.path.join(cls.get_test_data_dir(), *path)

    @classmethod
    def get_data_file(cls, index=1) -> File:
        return File(cls.get_test_data_path('dataset'+str(index)+'.csv'))

    @classmethod
    def get_dataset(cls, index=1, header=None, targets=[]) -> Table:
        return TableImporter.call(cls.get_data_file(index=index), {
            "delimiter": ",",
            "header": header,
            #"targets": targets,
            "metadata_columns": [{
                "column": k,
                "keep_in_data": True,
                "is_target": True
            } for k in targets]
        })
