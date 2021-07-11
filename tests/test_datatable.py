import os
import asyncio
import unittest

from gaia.datatable import Datatable, Importer as DatatableImporter
from gws.settings import Settings
from gws.protocol import Protocol
from gws.unittest import GTest

class TestImporter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        GTest.drop_tables()
        GTest.create_tables()
        GTest.init()
        
    @classmethod
    def tearDownClass(cls):
        GTest.drop_tables()
        
    def test_importer(self):
        p0 = DatatableImporter(instance_name="p0")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)

        def _on_end(*args, **kwargs):  
            ds = p0.output['datatable']
            self.assertEquals(ds.nb_columns, 5)
            self.assertEquals(ds.nb_rows, 150)
            self.assertEquals(ds.table.values[0,0], 5.1)
            self.assertEquals(ds.table.values[0,1], 3.5)
            self.assertEquals(ds.table.values[149,0], 5.9)
            
            self.assertEquals(list(ds.column_names), ["sepal.length","sepal.width","petal.length","petal.width","variety"])
            print(ds)
            
        e = p0.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_on_end)
        asyncio.run( e.run() )
        
    def test_importer_no_head(self):
        p0 = DatatableImporter(instance_name="p0")
        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        p0.set_param("file_path", os.path.join(test_dir, "./iris_no_head.csv"))
        p0.set_param("delimiter", ",")
            
        def _on_end(*args, **kwargs):  
            ds = p0.output['datatable']
            
            self.assertEquals(ds.nb_columns, 5)
            self.assertEquals(ds.nb_rows, 150)
            self.assertEquals(ds.table.values[0,0], 5.1)
            self.assertEquals(ds.table.values[0,1], 3.5)
            self.assertEquals(ds.table.values[149,0], 5.9)

            self.assertEquals(list(ds.column_names), list(range(0,5)))
            print(ds)
            
        e = p0.create_experiment(study=GTest.study, user=GTest.user)
        e.on_end(_on_end)
        asyncio.run( e.run() )

        