
import os
import asyncio
import unittest

from gaia.datatable import Datatable, Importer as DatatableImporter
from gws.settings import Settings

class TestImporter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        Datatable.drop_table()

    def test_importer(self):
        
        async def _import_iris(self):
            p0 = DatatableImporter()
            settings = Settings.retrieve()

            test_dir = settings.get_dir("gaia:testdata_dir")
            p0.set_param("file_path", os.path.join(test_dir, "./iris.csv"))
            p0.set_param("delimiter", ",")
            p0.set_param("header", 0)
            await p0.run()

            ds = p0.output['datatable']
            self.assertEquals(ds.nb_columns, 5)
            self.assertEquals(ds.nb_rows, 150)
            self.assertEquals(ds.table.values[0,0], 5.1)
            self.assertEquals(ds.table.values[0,1], 3.5)
            self.assertEquals(ds.table.values[149,0], 5.9)
            
            self.assertEquals(list(ds.column_names), ["sepal.length","sepal.width","petal.length","petal.width","variety"])
            print(ds)
            
        asyncio.run( _import_iris(self) )
        
    def test_importer_no_head(self):

        async def _import_iris_no_head(self):
            p0 = DatatableImporter()
            settings = Settings.retrieve()

            test_dir = settings.get_dir("gaia:testdata_dir")
            p0.set_param("file_path", os.path.join(test_dir, "./iris_no_head.csv"))
            p0.set_param("delimiter", ",")
            await p0.run()

            ds = p0.output['datatable']

            self.assertEquals(ds.nb_columns, 5)
            self.assertEquals(ds.nb_rows, 150)
            self.assertEquals(ds.table.values[0,0], 5.1)
            self.assertEquals(ds.table.values[0,1], 3.5)
            self.assertEquals(ds.table.values[149,0], 5.9)

            self.assertEquals(list(ds.column_names), list(range(0,5)))
            print(ds)
            
        asyncio.run( _import_iris_no_head(self) )

        