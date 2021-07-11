# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import pandas
from pandas import DataFrame

from gws.process import Process
from gws.resource import Resource
from gws.logger import Error

#====================================================================================================================
#====================================================================================================================

class Dataset(Resource):
    _features: DataFrame = None
    _targets: DataFrame = None
    _store_file_name = 'data.pkl'

    def __init__(self, *args, features: DataFrame = None, targets: DataFrame = None, **kwargs):
        if features is None:
            features = DataFrame()
        
        if targets is None:
            targets = DataFrame()

        self._features = features
        self._targets = targets
        super().__init__(*args, **kwargs)

    # -- C --

    @property
    def feature_names(self) -> list:
        """ 
        Returns the feaures names of the Dataset.

        :return: The list of feature names or `None` is no feature names exist
        :rtype: list or None
        """
        try:
            return self._features.columns.values.tolist()
        except:
            return None

    def feature_exists(self, name) -> bool:
        return name in self.feature_names

    @property
    def target_names(self) -> list:
        """ 
        Returns the target names.

        :return: The list of target names or `None` is no target names exist
        :rtype: list or None
        """
        try:
            return self._targets.columns.values.tolist()
        except:
            return None

    def target_exists(self, name) -> bool:
        return name in self.target_names

    # -- D --

    @property
    def features(self) -> DataFrame:
        """ 
        Returns the inner DataFrame.Alias of :property:`Dataset.features`.

        :return: The inner DataFrame
        :rtype: pandas.DataFrame
        """
        return self._features

    @property
    def targets(self) -> DataFrame:
        """ 
        Returns the inner DataFrame.Alias of :property:`Dataset.targets`.

        :return: The inner DataFrame
        :rtype: pandas.DataFrame
        """
        return self._targets

    # -- H --

    def head(self, n=5) -> (DataFrame, DataFrame):
        """ 
        Returns the first n rows for the features ant targets.

        :param n: Number of rows
        :param n: int
        :return: Two `panda.DataFrame` objects representing the n first rows of the `features` and `targets`
        :rtype: tuple, (pandas.DataFrame, pandas.DataFrame)
        """

        f = self._features.head(n)

        if self._target is None:
            t = None
        else:
            t = self._targets.head(n)

        return f, t

    # -- I --

    @property
    def instance_names(self) -> list:
        """ 
        Returns the instance names.

        :return: The list of instance names
        :rtype: list
        """
        return self._features.index.values.tolist()

    # -- N --

    @property
    def nb_features(self) -> int:
        """ 
        Returns the number of features.

        :return: The number of features 
        :rtype: int
        """
        return self._features.shape[1]
    
    @property
    def nb_instances(self) -> int:
        """ 
        Returns the number of instances.

        :return: The number of instances 
        :rtype: int
        """
        return self._features.shape[0]

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
        
        return f"Features: \n{self._features.__str__()} \n\nTargets: \n{self._targets.__str__()} "
    
        #n = self.nb_instances
        #if n > 10:
        #    
        #    x = self._features.iloc[:5,:]
        #    y = self._targets if self._targets.empty else self._targets.iloc[:5,:]
        #    t1 = pandas.concat( [x, y], axis=1 )
 
        #    x = self._features.iloc[5,:]
        #    y = self._targets if self._targets.empty else self._targets.iloc[5,:]
        #    t_i = pandas.concat( [x, y], axis=1 )
        #    t_i.iloc[:,:] = '...'
        #    t_i.index = pandas.Index(['..'])
            
        #    x = self._features.iloc[n-5:n,:]
        #    y = self._targets if self._targets.empty else self._targets.iloc[n-5:n,:]
        #    t2 = pandas.concat( [x,y], axis=1 )
        #    t = pandas.concat([t1,t_i,t2], axis=0)
        #else:
        #    t = pandas.concat([self._features,self._targets], axis=1)
        #
        #return t.__str__()

    # -- V --

    # -- W --

#====================================================================================================================
#====================================================================================================================

class Importer(Process):
    input_specs = {}
    output_specs = {'dataset': Dataset}
    config_specs = {
        'file_path': {"type": 'str', "default": ""},
        'delimiter': {"type": 'str', "default": '\t', "description": "Delimiter character. Only for parsing CSV files"},
        'header': {"type": 'int', "default": None, "description": "Row number to use as the column names. Use None to prevent parsing column names. Only for parsing CSV files"},
        'index' : {"type": 'int', "default": None, "description": "Column number to use as the row names. Use None to prevent parsing row names. Only for parsing CSV files"},
        'targets': {"type": 'list', "default": '[]', "description": "List of integers or strings (eg. ['name', 6, '7'])"},
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
            raise Error("Importer", "task", "Cannot detect the file type using file extension. Valid file extensions are [.xls, .xlsx, .csv, .tsv, .txt, .tab].")
        
        out_type = self.output_specs["dataset"]

        if self.get_param('targets') == "":
            ds = out_type(features=df)
        else:
            targets = self.get_param('targets')
            try:
                t_df = df.loc[:,targets]
            except:
                raise Error("Importer", "task", f"The targets {targets} are no found in column names. Please check targets names or set parameter 'header' to read column names.")
            
            df.drop(columns = targets, inplace = True)
            ds = out_type(features = df, targets = t_df)

        self.output['dataset'] = ds

#====================================================================================================================
#====================================================================================================================

class Exporter(Process):
    input_specs = {'dataset': Dataset}
    output_specs = {}
    config_specs = {
        'file_path': {"type": 'str', "default": ""},
        'delimiter': {"type": 'str', "default": "\t", "description": "Delimiter character. Only for parsing CSV files"},
        'header': {"type": 'int', "default": 0, "description": "Row number(s) to use as the column names, and the start of the data. Only for parsing CSV files"},
    }

    async def task(self):
        pass