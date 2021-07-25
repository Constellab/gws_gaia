# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import pandas
from pandas import DataFrame

from gws.process import Process
from gws.resource import Resource
from gws.csv import CSVData
from gws.exception.bad_request_exception import BadRequestException

#====================================================================================================================
#====================================================================================================================

class Datatable(CSVData):
    """
    Datatable class
    """
    pass

#====================================================================================================================
#====================================================================================================================

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
            raise BadRequestException("Cannot detect the file type using file extension. Valid file extensions are [.xls, .xlsx, .csv, .tsv, .txt, .tab].")
        
        t = self.output_specs["datatable"]
        self.output['datatable'] = t(table=df)

#====================================================================================================================
#====================================================================================================================

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