# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import pandas
from pandas import DataFrame

from gws.model import Process, Config
from gws.model import Resource
from gws.controller import Controller
from gws.logger import Logger

class Datatable(Resource):
    _table: DataFrame = None
    _store_file_name = 'data.pkl'

    def __init__(self, table: DataFrame = None, *args, **kwargs):
        self._table = table
        super().__init__(*args, **kwargs)

    # -- C --

    @property
    def column_names(self) -> list:
        """ 
        Returns the column names of the Datatable.

        :return: The list of column names or `None` is no column names exist
        :rtype: list or None
        """
        try:
            return self._table.columns.values.tolist()
        except:
            return None

    def column_exists(self, name) -> bool:
        return name in self.column_names

    # -- D --

    @property
    def table(self) -> DataFrame:
        """ 
        Returns the inner DataFrame.Alias of :property:`Datatable.table`.

        :return: The inner DataFrame
        :rtype: pandas.DataFrame
        """
        return self._table

    # -- H --

    def head(self, n=5) -> DataFrame:
        """ 
        Returns the first n rows for the columns ant targets.

        :param n: Number of rows
        :param n: int
        :return: The `panda.DataFrame` objects representing the n first rows of the `table`
        :rtype: pandas.DataFrame
        """

        return self._table.head(n)

    # -- I --

    @property
    def row_names(self) -> list:
        """ 
        Returns the row names.

        :return: The list of row names
        :rtype: list
        """
        return self._table.index.values.tolist()

    # -- N --

    @property
    def nb_columns(self) -> int:
        """ 
        Returns the number of columns.

        :return: The number of columns 
        :rtype: int
        """
        return self._table.shape[1]
    
    @property
    def nb_rows(self) -> int:
        """ 
        Returns the number of rows.

        :return: The number of rows 
        :rtype: int
        """
        return self._table.shape[0]

    @property
    def nb_targets(self) -> int:
        """ 
        Returns the number of targets.

        :return: The number of targets (0 is no targets exist)
        :rtype: int
        """
        if self._targets is None:
            return 0
        else:
            return self._targets.shape[1]

    # -- R --

    # -- S --

    def __str__(self):
        return self._table.__str__()

    # -- V --

    # -- W --

class Importer(Process):
    input_specs = {}
    output_specs = {'datatable': Datatable}
    config_specs = {
        'file_path': {"type": 'str', "default": ""},
        'delimiter': {"type": 'str', "default": '\t', "description": "Delimiter character. Only for parsing CSV files"},
        'header': {"type": 'int', "default": None, "description": "Row number to use as the column names. Use None to prevent parsing column names. Only for parsing CSV files"},
        'index' : {"type": 'int', "default": None, "description": "Column number to use as the row names. Use None to prevent parsing row names. Only for parsing CSV files"},
    }
    
    async def task(self):
        file_path = self.get_param("file_path")
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".xls", ".xlsx"]:
            df = pandas.read_excel(file_path)
        elif file_extension in [".csv", ".tsv", ".txt", ".tab"]:
            df = pandas.read_table(
                file_path, 
                sep = self.get_param("delimiter"), 
                header = self.get_param("header"),
                index_col = self.get_param("index")
            )
        else:
            Logger.error(Exception("Importer", "task", "Cannot detect the file type using file extension. Valid file extensions are [.xls, .xlsx, .csv, .tsv, .txt, .tab]."))
        
        t = self.output_specs["datatable"]
        self.output['datatable'] = t(table=df)


class Exporter(Process):
    input_specs = {'datatable': Datatable}
    output_specs = {}
    config_specs = {
        'file_path': {"type": 'str', "default": ""},
        'delimiter': {"type": 'str', "default": "\t", "description": "Delimiter character. Only for parsing CSV files"},
        'header': {"type": 'int', "default": 0, "description": "Row number(s) to use as the column names, and the start of the data. Only for parsing CSV files"},
    }

    async def task(self):
        pass