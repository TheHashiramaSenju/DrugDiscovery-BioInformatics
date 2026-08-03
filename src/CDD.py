import pandas as pd 
import numpy as np
import sqlite3
import duckdb
from chembl_webresource_client.new_client import new_client
from abc import ABC, abstractmethod
from typing import Optional
import polars as pl 
from pathlib import Path
import io

ROOT_FOLDER_INTAKE =input("The absolute path of the root folder")
ROOT_FOLDER =  Path(ROOT_FOLDER_INTAKE)

def data_retrieval_desc(target_name: str) -> pd.DataFrame:
    
    if 'new_client' not in globals():
        raise NameError("New_client is not defined. Please import or initialize it")
    
    #converting into raw strings
    target_name_str = str(target_name)
    
    try:
        target = new_client.target
        target_query = target.search(target_name_str)
        
        if isinstance(target_query, list) and len(target_query) > 0:
            target_query.Dataframe.from_dict(target_query)
        else:
            targets = pd.DataFrame()
    
    except Exception as e:
        raise RuntimeError("Failed to initialize target client") from e

    print(f"Target data retrieval for Target ID: {target_name} \n {targets}")
    
    return targets

    
def select_target(target_index:int, targets: pd.DataFrame = None) -> List:

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
    
    selected_target = targets['target_chembl_id'].iloc[target_index]
    info_of_chembl = targets.loc[targets['target_chembl_id'] == selected_target, ["pref_name", "organism"]]
    
    flattened = info_of_chembl.values.flatten().to_list()
    consolidate = [selected_target, flattened]
    
    return consolidate

class SQLEngine(ABC):
    
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
class DuckDBEngine(SQLEngine): 
    
    def __init__(self, connection = None, targeted = None):
        self.conn = connection if connection else duckdb.connect()
        self.targeted = targeted if targeted else [None, [None, None]]
        #table-info
        self.targetindex = targeted[0]
        self.pref_name, self.organism = targeted[1]
    
    @classmethod #fun-concept --> decorator
    def rootfolder(cls):
        '''
        #this gives you the entire path from the root
             current_dir = os.getcwd()
                return current_dir
        '''      
        #getting the current directory
        
        current_file_location = Path(__file__).resolve()
        relative_path = current_file_location.relative_to(ROOT_FOLDER)
        levels = len(relative_path) - 1
        current_file_location.parents[levels]
        
        return current_file_location        
    
    @classmethod
    def current_folder(cls):
        
        '''
        This gives the current file's path. 
        '''
        currentfile = Path(__file__).resolve()
        return currentfile
            
        
    def create_and_load_csv(self):
        
        # DuckDB can query CSV directly, but we register it as a table for consistency
        
        csv_folder_path = ROOT_FOLDER / "database" / "csv"
        csv_folder_path.makedir(parent=True, exist_ok = True)
        
        if not self.targetindex:
            raise ValueError("Target CHEMBL_ID is missing from engine configuration")
            
        activity = new_client.activity
        
        res = activity.filter(
            target_chembl_id=self.targetindex,
            standard_value__isnull=False,
            standard_type__in=["IC50", "EC50", "Ki", "Kd" ]
        )
        
        csv_folder_path = ROOT_FOLDER / "database" / "csv"
        
        #Dictionary to DataFrame
        df = pd.DataFrame.from_dict(res) 

        #Filename for saving
        file_name = csv_folder_path / f"{self.pref_name}_{self.organism}.csv"
        
        #Dataframe - CSV
        df.to_csv(file_name, index=False)
        

    def create_database(self, csv_filepath:str, table_name:str):
        
                
        #The CSV file is already made so we can proceed with the database conversion
        
        
    def query_check(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def get_columns(self) -> list:
        return self._columns
    

    def close(self):
        self.con.close()

def data_exploration():
    
    '''
    Made to actually explore data and gether insights about the data and its nature
    Can also be seen in Data-Wrangler, yet this seems to build more intuition 
    '''
    