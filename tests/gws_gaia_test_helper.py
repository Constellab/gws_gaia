

import os

from gws_core import File, Settings, Table, TableImporter


class GWSGaiaTestHelper():

    @classmethod
    def get_test_data_dir(cls) -> str:
        return Settings.get_instance().get_variable("gws_gaia:testdata_dir")

    @classmethod
    def get_test_data_path(cls, *path: str) -> str:
        return os.path.join(cls.get_test_data_dir(), *path)

    @classmethod
    def get_data_file(cls, index=1) -> File:
        return File(cls.get_test_data_path('dataset'+str(index)+'.csv'))

    @classmethod
    def get_table(cls, index=1, header=None, targets=[]) -> Table:
        return TableImporter.call(cls.get_data_file(index=index), {
            "delimiter": ",",
            "header": header,
            "metadata_columns": [{
                "column": k,
                "keep_in_table": True,
            } for k in targets]
        })
