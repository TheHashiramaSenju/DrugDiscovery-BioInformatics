import pandas as pd 
import numpy as np
import sqlite3
import duckdb
from chembl_webresource_client.new_client import new_client
from abc import ABC, abstractmethod
from typing import Optional
import polars as pl 
import os 



def data_retrieval_desc(target_name: str) -> pd.DataFrame:
    
    if 'new_client' not in globals():
        raise NameError("New_client is not defined. Please import or initialize it")
    
    target = new_client.target
    target_query = target.search(target_name) #searches for something like "corona_virus"
    
    if isinstance(target_query, list) and len(target_query) > 0:
        targets = pd.DataFrame.from_dict(target_query)
    else:
        targets = pd.DataFrame()
    
    print(f"Target data retrieval for Target ID: {target_name} \n {targets}")
    
    return targets

    
    
def select_target(target_index:int, targets: pd.DataFrame = None) -> pd.DataFrame:

    if not isinstance(target_index, int):
        raise TypeError("target_index must be an integer")
    
    if targets is None:
        
        raise ValueError("You must provide either a 'targets' DataFrame or a 'target_id' to fetch.")
    else:
        targets = data_retrieval_desc(target_index)
    
    if targets.empty:
        raise ValueError("The targets DataFrame is empty. Cannot select an index.")
    
    if 'target_chembl_id' not in targets.columns:
        raise KeyError("Column 'target_chembl_id' not found in targets DataFrame.")
    
    selected_target = targets.target_chembl_id[target_index] 
    
    #now we take a dataframe in a dataframe and return a dimension of it, this will be in the (n-1) dimension 
    return selected_target #now the entire DataFrame is returned and we can use it for further processing.


class SQLEngine(ABC): #does this type of inheritance on python work?
    
    """
    Abstract classes usually defines what each must actually do 
    Here - We are using ELT - Extract, Load, Transform methods for loading data into the database and then querying it. 
    """
    
    @abstractmethod
    def csv_conversion(self,):
        pass 
    
    @abstractmethod 
    #we have also included the loading of CSV in this same method, implemeneted to all the classes
    def create_database(self, filepath:str):
        pass 
    
    #transformation - this is a placeholder for any transformation that might be needed before querying the data.
    def transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass 
    
    @abstractmethod
    def query_check(self, sql:str) -> pd.DataFrame:
        pass 
    
    @abstractmethod
    def get_columns(self) -> list:
        pass
    


    

#TheSQL engine selector

class SQLiteEngine(SQLEngine):
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.table_name = "ReplicasePolyprotein"
        self.columns = []
        self.db_filename = "ReplicasePolyprotein.db"
    
    def load_csv(self, filepath:str):
        df = pd.read_csv(filepath)
        self._columns = df.columns.tolist()
        df.to_sql(self.table_name, self.conn, index=False, if_exists='replace')
    
    def create_database(self, filepath):
        return super().create_database(filepath)
        
    def query_check(self, sql:str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn)
    
    def get_columns(self):
        return self._columns
    
    def close(self):
        self.conn.close()
        
        
class DuckDBEngine(SQLEngine): 
    
    def __init__(self):
        self.con = duckdb.connect()
        self.table_name = "data"
        self._columns = []
        self.selected_target_id = self.SELECTED_ID # why did we use self here 
        self.db_filename = self.db_filename
        
    @classmethod #fun-concept --> decorator
    def filename(cls):
        
        '''
        #this gives you the entire path from the root
             current_dir = os.getcwd()
                return current_dir
        '''      
        #current_directory 
        absolute_path = os.getcwd()
        current_folder = os.path.basename(absolute_path)
        
        
        
    def load_csv(self, filepath: str):
        
        # DuckDB can query CSV directly, but we register it as a table for consistency
        
        self.con.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
        res = self.con.execute(f"DESCRIBE {self.table_name}").fetchall()
        self._columns = [col[0] for col in res]

    def create_database(self, csv_filepath:str, table_name:str):
        activity = new_client.activity
        res = activity.filter(
            target_chembl_id=self.selected_target_id,
            standard_value__isnull=False,
            standard_type__in=["IC50", "EC50", "Ki", "Kd" ]
        )
        
        #folder creation for storing db files
        os.makedirs('/database', exist_ok=True)

        #we use with statements here for connection basing. 
        
        with duckdb.connect(self.db_filename) as con:
            query = f"CREATE TABLE IF NOT EXISTS {self.table_name} AS SELECT * FROM read_csv_auto('{csv_filepath}');"
            conn.execute(query)
        
        print(f"Database created at {}")
            
        
        #The CSV file is already made so we can proceed with the database conversion
        
        
    def query_check(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def get_columns(self) -> list:
        return self._columns
    

    def close(self):
        self.con.close()
    
    
    
class PolarsEngine(SQLEngine):
    def __init__(self):
        self.ctx = None
        self._columns = []

    def load_csv(self, filepath: str):
        
        # Polars uses LazyFrames for efficiency
        lf = pl.scan_csv(filepath)
        self._columns = lf.collect_schema().names()
        
        # Create SQL Context and register the LazyFrame
        self.ctx = pl.SQLContext(register_globals=False, eager=False)
        self.ctx.register("data", lf)
        
    def create_database(self, filepath):
            return super().create_database(filepath)

    def query_check(self, sql: str) -> pd.DataFrame:
        # Execute and convert to Pandas for consistent output
        return self.ctx.execute(sql).collect().to_pandas()

    def get_columns(self) -> list:
        return self._columns
    
    def close(self):
        if self.ctx:
            self.ctx.close()
            
    

def data_exploration():
    
    '''
    Made to actually explore data and gether insights about the data and its nature
    Can also be seen in Data-Wrangler, yet this seems to build more intuition 
    '''
    