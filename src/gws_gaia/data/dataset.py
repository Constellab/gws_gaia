# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import numpy as np
import pandas
from pandas import DataFrame
from typing import Union, List
from pathlib import Path

from gws_core import (task_decorator, 
                        resource_decorator, BadRequestException, 
                        Table, File, TableExporter, TableImporter, TableLoader, TableDumper, 
                        StrParam, IntParam, ListParam, BoolParam, DataFrameRField)

#====================================================================================================================
#====================================================================================================================

@resource_decorator("Dataset")
class Dataset(Table):
    """
    Dataset class
    """
    
    features: DataFrame = DataFrameRField()
    targets: DataFrame = DataFrameRField()

    def __init__(self, *args, features: Union[DataFrame, np.ndarray] = None, 
                    targets: Union[DataFrame, np.ndarray] = None, 
                    feature_names: List[str]=None, target_names: List[str]=None, row_names: List[str]=None, **kwargs):
        super().__init__(*args, **kwargs)

        if features is not None:
            if isinstance(features, DataFrame):
                # OK!
                pass
            elif isinstance(features, (np.ndarray, list)):
                features = DataFrame(features)
                if feature_names:
                    features.columns = feature_names
                if row_names:
                    features.index = row_names
            else:
                raise BadRequestException(
                    "The table mus be an instance of DataFrame or Numpy array")
            self.features = features

        if targets is not None:
            if isinstance(targets, DataFrame):
                # OK!
                pass
            elif isinstance(targets, (np.ndarray, list)):
                targets = DataFrame(targets)
                if target_names:
                    targets.columns = target_names
                if row_names:
                    targets.index = row_names
            else:
                raise BadRequestException(
                    "The table mus be an instance of DataFrame or Numpy array")
            self.targets = targets

    # -- C --

    # -- E --

    def export_to_path(self, file_path: str, delimiter: str = "\t", index=True, file_format: str = None, **kwargs):
        """
        Export to a repository

        :param file_path: The destination file path
        :type file_path: File
        """

        file_extension = Path(file_path).suffix
        if file_extension in [".xls", ".xlsx"] or file_format in [".xls", ".xlsx"]:
            table = pandas.concat([self.features, self.targets])
            table.to_excel(file_path)
        elif file_extension in [".csv", ".tsv", ".txt", ".tab"] or file_format in [".csv", ".tsv", ".txt", ".tab"]:
            table = pandas.concat([self.features, self.targets])
            table.to_csv(
                file_path,
                sep=delimiter,
                index=index
            )
        else:
            raise BadRequestException(
                "Cannot detect the file type using file extension. Valid file extensions are [.xls, .xlsx, .csv, .tsv, .txt, .tab].")


    # -- F --

    @property
    def feature_names(self) -> list:
        """ 
        Returns the feaures names of the Dataset.

        :return: The list of feature names or `None` is no feature names exist
        :rtype: list or None
        """
        try:
            return self.features.columns.values.tolist()
        except:
            return None

    def feature_exists(self, name) -> bool:
        return name in self.feature_names

    # -- H --

    def head(self, n=5) -> (DataFrame, DataFrame):
        """ 
        Returns the first n rows for the features ant targets.

        :param n: Number of rows
        :param n: int
        :return: Two `panda.DataFrame` objects representing the n first rows of the `features` and `targets`
        :rtype: tuple, (pandas.DataFrame, pandas.DataFrame)
        """

        f = self.features.head(n)

        if self._target is None:
            t = None
        else:
            t = self.targets.head(n)

        return f, t

    # -- I --

    @classmethod
    def import_from_path(cls, file_path: str, delimiter: str = "\t", header=0, index_col=None, file_format: str = None, targets: list=None, **kwargs) -> 'Dataset':
        """
        Import from a repository

        :param file_path: The source file path
        :type file_path: file path
        :returns: the parsed data
        :rtype any
        """

        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".xls", ".xlsx"]:
            df = pandas.read_excel(file_path)
        elif file_extension in [".csv", ".tsv", ".txt", ".tab"]:
            df = pandas.read_table(
                file_path, 
                sep = delimiter,
                header = header,
                index_col = index_col
            )
        else:
            raise BadRequestException("Cannot detect the file type using file extension. Valid file extensions are [.xls, .xlsx, .csv, .tsv, .txt, .tab].")
        
        if not targets:
            ds = cls(features=df)
        else:
            try:
                t_df = df.loc[:,targets]
            except Exception as err:
                raise BadRequestException(f"The targets {targets} are no found in column names. Please check targets names or set parameter 'header' to read column names.") from err
            df.drop(columns = targets, inplace = True)
            ds = cls(features = df, targets = t_df)
        return ds

    @property
    def instance_names(self) -> list:
        """ 
        Returns the instance names.

        :return: The list of instance names
        :rtype: list
        """
        return self.features.index.values.tolist()

    # -- N --

    @property
    def nb_features(self) -> int:
        """ 
        Returns the number of features.

        :return: The number of features 
        :rtype: int
        """
        return self.features.shape[1]
    
    @property
    def nb_instances(self) -> int:
        """ 
        Returns the number of instances.

        :return: The number of instances 
        :rtype: int
        """
        return self.features.shape[0]

    @property
    def nb_targets(self) -> int:
        """ 
        Returns the number of targets.
        
        :return: The number of targets (0 is no targets exist)
        :rtype: int
        """
        if self.targets is None:
            return 0
        else:
            return self.targets.shape[1]

    # -- R --

    @property
    def row_names(self) -> list:
        """ 
        Alias of :property:`instance_names`
        """
        
        return self.instance_names

    # -- S --

    def __str__(self):        
        return f"Features: \n{self.features.__str__()} \n\nTargets: \n{self.targets.__str__()} "

    # -- T --

    @property
    def target_names(self) -> list:
        """ 
        Returns the target names.

        :return: The list of target names or `None` is no target names exist
        :rtype: list or None
        """
        try:
            return self.targets.columns.values.tolist()
        except:
            return None

    def target_exists(self, name) -> bool:
        return name in self.target_names

    # -- W --

#====================================================================================================================
#====================================================================================================================

@task_decorator("DatasetImporter")
class DatasetImporter(TableImporter):
    input_specs = {'file': File}
    output_specs = {'dataset': Dataset}
    config_specs = {
        'file_format': StrParam(default_value=".csv", short_description="File format"),
        'delimiter': StrParam(default_value='\t', short_description="Delimiter character. Only for parsing CSV files"),
        'header': IntParam(optional=True, default_value=0, short_description="Row number to use as the column names. Use None to prevent parsing column names. Only for parsing CSV files"),
        'index' : IntParam(optional=True, short_description="Column number to use as the row names. Use None to prevent parsing row names. Only for parsing CSV files"),
        'targets': ListParam(default_value='[]', short_description="List of integers or strings (eg. ['name', 6, '7'])"),
    }

@task_decorator("DatasetExporter")
class DatasetExporter(TableExporter):
    input_specs = {'dataset': Dataset}
    output_specs = {'file': File}
    config_specs = {
        'file_format': StrParam(default_value=".csv", short_description="File format"),
        'delimiter': StrParam(default_value="\t", short_description="Delimiter character. Only for parsing CSV files"),
        'header': BoolParam(optional=True, short_description= "Write column names (header)"),
        'index': BoolParam(optional=True, short_description="Write row names (index)"),
    }

#====================================================================================================================
#====================================================================================================================

@task_decorator("DatasetLoader")
class DatasetLoader(TableLoader):
    input_specs = {}
    output_specs = {'dataset': Dataset}
    config_specs = {
        'file_path': StrParam(short_description="File path"),
        'file_format': StrParam(default_value=".csv", short_description="File format"),
        'delimiter': StrParam(default_value='\t', short_description="Delimiter character. Only for parsing CSV files"),
        'header': IntParam(optional=True, default_value=0, short_description="Row number to use as the column names. Use None to prevent parsing column names. Only for parsing CSV files"),
        'index' : IntParam(optional=True, short_description="Column number to use as the row names. Use None to prevent parsing row names. Only for parsing CSV files"),
        'targets': ListParam(default_value='[]', short_description="List of integers or strings (eg. ['name', 6, '7'])"),
    }

@task_decorator("DatasetDumper")
class DatasetDumper(TableDumper):
    input_specs = {'dataset': Dataset}
    output_specs = {}
    config_specs = {
        'file_path': StrParam(short_description="File path"),
        'file_format': StrParam(default_value=".csv", short_description="File format"),
        'delimiter': StrParam(default_value="\t", short_description="Delimiter character. Only for parsing CSV files"),
        'header': BoolParam(optional=True, short_description= "Write column names (header)"),
        'index': BoolParam(optional=True, short_description="Write row names (index)"),
    }