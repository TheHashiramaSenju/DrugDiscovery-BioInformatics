from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Optional
import numpy as np
from chembl_webresource_client.new_client import new_client
import duckdb
import pandas as pd
from rdkit import Chem 
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.stats import median_abs_deviation
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder


def data_retrieval_desc(target_name: str) -> pd.DataFrame:
    if not target_name or not target_name.strip():
        raise ValueError("Target name cannot be empty.")

    try:
        target = new_client.target
        target_query = target.search(str(target_name).strip())
        targets = pd.DataFrame.from_dict(target_query)
    except Exception as e:
        raise RuntimeError("Failed to retrieve target data from ChEMBL.") from e

    if targets.empty:
         raise ValueError(f"No targets found matching: '{target_name}'")

    display_cols = [c for c in ["target_chembl_id", "pref_name", "organism", "target_type"] if c in targets.columns]
    print("\nTop 10 Target Matches:")
    print(targets[display_cols].head(10))

    return targets


def select_target(target_index: int, targets: pd.DataFrame) -> list:
    
    if not isinstance(target_index, int):
        raise TypeError("target_index must be an integer.")

    if targets is None or targets.empty:
        raise ValueError("The targets DataFrame is empty or None.")

    if "target_chembl_id" not in targets.columns:
        raise KeyError("Column 'target_chembl_id' not found in targets DataFrame.")

    if target_index < 0 or target_index >= len(targets):
        raise IndexError(f"target_index {target_index} is out of bounds (0 to {len(targets) - 1}).")

    selected_row = targets.iloc[target_index]
    
    selected_target = str(selected_row["target_chembl_id"])
    pref_name = str(selected_row.get("pref_name", "UnknownTarget"))
    organism = str(selected_row.get("organism", "UnknownOrganism"))

    clean_name = re.sub(r"\W+", "_", pref_name).strip("_")
    clean_org = re.sub(r"\W+", "_", organism).strip("_")

    return [selected_target, [clean_name, clean_org]]


class SQLEngine(ABC):
    
    @abstractmethod
    def create_and_load_csv(self) -> tuple[Path, str]:
        pass

    @abstractmethod
    def create_database(self, csv_filepath: Path, table_name: str) -> None:
        pass

    @abstractmethod
    def query_check(self, table_name: str) -> pd.DataFrame:
        pass


class DuckDBEngine(SQLEngine):
    
    def __init__(self, connection: Optional[duckdb.DuckDBPyConnection], targeted: list, root_folder: Path):
        self.conn = connection if connection else duckdb.connect() 
        self.target_chembl_id = targeted[0]
        self.pref_name, self.organism = targeted[1]
        self.root_folder = Path(root_folder)

        if not self.target_chembl_id:
            raise ValueError("Target CHEMBL ID is missing from configuration.")

    def create_and_load_csv(self) -> tuple[Path, str]:
        csv_folder = self.root_folder / "database" / "csv"
        csv_folder.mkdir(parents=True, exist_ok=True)

        table_name = f"{self.pref_name}_{self.organism}"
        csv_file_path = csv_folder / f"{table_name}.csv"

        print(f"\nFetching complete IC50 activity table for {self.target_chembl_id} ({table_name})...")
        print("Network transfer started. This may take a few minutes for large targets.")

        activity = new_client.activity
        res = activity.filter(
            target_chembl_id=self.target_chembl_id,
            standard_value__isnull=False,
            standard_type__in=["IC50"]
        )

        records = []
        for i, record in enumerate(res, 1):
            records.append(record)
            if i % 1000 == 0:
                print(f"Successfully downloaded {i} records...")

        if not records:
            raise ValueError(f"No IC50 activity records found for {self.target_chembl_id}.")

        print(f"Finished downloading {len(records)} total records. Writing to CSV...")
        df = pd.DataFrame(records)
        df.to_csv(csv_file_path, index=False)
        print(f"Saved: {csv_file_path}")

        return csv_file_path, table_name

    def create_database(self, csv_filepath: Path, table_name: str) -> None:
        
        if not csv_filepath or not csv_filepath.exists():
            raise FileNotFoundError(f"Source CSV file not found: {csv_filepath}")

        db_folder = self.root_folder / "database" / "db"
        parquet_folder = db_folder / "parquet"
        db_folder.mkdir(parents=True, exist_ok=True)
        parquet_folder.mkdir(parents=True, exist_ok=True)

        safe_table = f'"{table_name}"'
        posix_csv = csv_filepath.as_posix()

        print("\nBuilding DuckDB tables and persistent storage")

        self.conn.execute(f"CREATE OR REPLACE TABLE {safe_table} AS SELECT * FROM read_csv_auto('{posix_csv}');")

        target_parquet = parquet_folder / f"{table_name}.parquet"
        self.conn.execute(f"COPY {safe_table} TO '{target_parquet.as_posix()}' (FORMAT PARQUET);")

        target_db = db_folder / f"{table_name}.db"
        self.conn.execute(f"ATTACH '{target_db.as_posix()}' AS disk_db;")
        self.conn.execute(f"CREATE OR REPLACE TABLE disk_db.{safe_table} AS SELECT * FROM {safe_table};")
        self.conn.execute("DETACH disk_db;")

        print(f"Parquet saved: {target_parquet}")
        print(f"DB File saved: {target_db}")

    def query_check(self, table_name: str) -> pd.DataFrame:
        print("\nRunning verification query (LIMIT 5):")
        safe_table = f'"{table_name}"'
        return self.conn.execute(f"SELECT * FROM {safe_table} LIMIT 5;").fetchdf()


class DataCleaning:
    
    remover = SaltRemover()
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.mismatch = []
        self.filename = self.path.name if self.path.exists() else "unknown_file"

    def catch_bad_row(self, bad_line):
        self.mismatch.append(bad_line)
        return None
    
    def loading_csv(self, path: Path) -> pd.DataFrame:
        print(f"\nLoading and parsing CSV file: {path.name}")
        dataset_clean = pd.read_csv(
            path, 
            engine='python', 
            on_bad_lines=self.catch_bad_row
        ) 
        if self.mismatch:
            print(f"Skipped {len(self.mismatch)} corrupted/mismatched rows.")
        return dataset_clean 
    
    def get_clean_filename(self) -> str:
        filename = self.filename
        print(filename)
        if filename.startswith('.'):
            filename = filename[1:]
        return filename.split('.')[0] #we handled null pointer exception in this case where we unindented the return statement to return something on 
            
    '''
    For pandas
    If you hand it a list of True/False values, it filters rows.
    If you hand it a list of names or numbers, it tries to fetch columns.
    
    '''
    #helper function for entire dataframe
    @staticmethod
    def stip_salt(smiles_string):
    
        if pd.isna(smiles_string):
            return None 
        
        mol = Chem.MolFromSmiles(smiles_string)
        
        if mol is None:
            return None
        
        #now we will remove those salts 
        stripped_mol = DataCleaning.remover.StripMol(mol)
        return Chem.MolToSmiles(stripped_mol)
    
    @staticmethod
    def InChIKeyConversion(smiles_string):
        
        if pd.isna(smiles_string):
            return None 
        
        mol = Chem.MolFromSmiles(smiles_string)
        
        if mol is None:
            return None 
    
        inchikey = Chem.MolToInchiKey(mol)
        
        return inchikey
        
        #NOTE : INCHI and INCHI key -> INCHI is too BIG and hence INCHI key we will be using for database wide comparisons and analysis        
        

        
    def null_and_columnhandler(self, path: Path) -> Path:
        
        dataset_clean = self.loading_csv(path)
        
        dataset_clean = dataset_clean.dropna(axis=1, how="all")
        
        cols_to_drop = ["qudt_units", "uo_units", "toid", "document_chembl_id", "_journal", "_year", 
                        "assay_descriptions", "activity_comment", "upper_value", 
                        "molecule_pref_name", "type", "units", "value","document_journal", "document_year", 
                        "assay_description", "activity_id", "activity_properties", "ligand_efficiency" ]
        
        existing_cols_to_drop = [col for col in cols_to_drop if col in dataset_clean.columns]
        
        if existing_cols_to_drop:
            dataset_clean = dataset_clean.drop(columns=existing_cols_to_drop)
            print(f"Dropped unneeded columns: {existing_cols_to_drop}")
        
        targets_data_validity_comment = ["Values appear to be an order of magnitude different from previously reported, so units may be incorrect", 
                                         "Potential transcription error"]
        
        targets_data_validity_desc = ["Values for this activity type are unusually large/small, so may not be accurate", 
                                      "Values appear to be an order of magnitude different from previously reported, so units may be incorrect"]
        
        mask_1 = dataset_clean["data_validity_comment"].isin(targets_data_validity_comment)
        mask_2 = dataset_clean["data_validity_description"].isin(targets_data_validity_desc)
        
        combined_bad_rows = mask_1 | mask_2        
    
        rows_to_drop = dataset_clean[combined_bad_rows].index
        dataset_clean = dataset_clean.drop(rows_to_drop, axis = 0)
        
        dataset_clean = dataset_clean.drop(columns=["data_validity_comment", "data_validity_description"], axis=1)
        
        clean_filename = self.get_clean_filename()
        output_path = path.parent / f"{clean_filename}_cleaned.csv"
        
        dataset_clean.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved successfully: {output_path}")
        
        return output_path

    
    def InChIstandardization(self) -> pd.DataFrame:
        # Notice how there is NO call to null_and_columnhandler here anymore!
        # This function strictly assumes it is receiving an already cleaned file.
        cleaned_csv_path: Path = self.null_and_columnhandler(self.path)
        df = pd.read_csv(cleaned_csv_path)
        df["cleaned_smiles"] = df["canonical_smiles"].apply(DataCleaning.stip_salt)
        
        df.drop("canonical_smiles", axis=1, inplace=True)
        
        #now creating a InChI key 
        df["InChIkey"] = df["cleaned_smiles"].apply(DataCleaning.InChIKeyConversion)
        
        df.to_csv(cleaned_csv_path, index=False)
        print(f"Standardized CSV saved successfully: {cleaned_csv_path}")
        
        return df  
        
    
    def basic_duplicate_resolution(self) ->pd.DataFrame:
        
        dataframe = self.InChIstandardization()
        #clearing already flagged duplicates --> Toll gate analogy
        dataframe = dataframe[dataframe["potential_duplicate"] != 1].copy()
        
        return dataframe

        
    '''
    Since, standard_value is 0% missing data. So we dont need
    special imputation techniques for values
    
    We will add in the columns to drop program in the null_column_handler part 
    '''
    
    def IC50_units_standardization(self) -> pd.DataFrame :
        
        dataframe = self.basic_duplicate_resolution()
        units = ["10'5pM", "10'6pM", "10'3pM", "nM"]
        
        dataframe = dataframe[dataframe["standard_units"].isin(units)].copy()
        
        conditions = [

            (dataframe["standard_units"] == "10'5pM"), 
            (dataframe["standard_units"] == "10'6pM"),
            (dataframe["standard_units"] == "10'3pM"),
            (dataframe["standard_units"] == "nM")
        ]
        
        choices = [
            #pico-molar is thousand times smaller than nano-molar (10 ** 3) and hence conversion factor will be 10 ** -3
            (dataframe["standard_value"] * 100),
            (dataframe["standard_value"] * 1000),
            (dataframe["standard_value"] * 1), 
            (dataframe["standard_value"] * 1) 
            
        ]
        
        dataframe["IC50"] = np.select(conditions, choices)
        dataframe["standard_units"] = "nM"
               
        return dataframe
          
    def IC50_to_PIC50Conv(self) -> pd.DataFrame: 
        
        dataframe = self.IC50_units_standardization()
        
        conditions = [
            (dataframe["standard_type"] == "IC50") & (dataframe["standard_value"] > 0),
            (dataframe["standard_type"] == "Log IC50"),
            (dataframe["standard_type"] == "pIC50"), 
            (dataframe["standard_type"] == "Log IC50(nM)")
        ]
        choices = [
            -1 * np.log10(dataframe["standard_value"] * 10 ** -9),
            -1 * dataframe["standard_value"],
            dataframe["standard_value"],
            -9 + dataframe["standard_value"]
        ]
        
        dataframe["PIC50"] = np.select(conditions, choices) #dropping instinct
        dataframe["PIC50"] = dataframe["PIC50"].round(5)
        #the conditions are not getting applied 
        
        return dataframe        
        
    def InChI_based_duplicate_res(self) -> pd.DataFrame:
        
        #now there are non-distinct InChI values, they might be from multiple tests so here we address those specific set of values 
        #mean-absolute-deviation
        dataframe = self.IC50_to_PIC50Conv()
        df = dataframe.copy()
        df["median"] = df.groupby("InChIkey")["PIC50"].transform("median")
        df["compared_median"] = abs(df["median"] - df["PIC50"])
        
        df["group_MAD"] = df.groupby("InChIkey")["compared_median"].transform("median") #masking function taking place 
        
        mad_threshold: float = 1.0 #reserch more about this values.
        cleaned_df = df[df["group_MAD"] <= mad_threshold].copy()
        cleaned_df["PIC50"] = cleaned_df["median"]
        
        final_df = cleaned_df.drop_duplicates(subset=["InChIkey", "cleaned_smiles", "PIC50"]).copy()
        final_df = final_df.drop(columns = ["median", "compared_median", "group_MAD"])
        
        return final_df

    def relationalvalue(self) -> pd.DataFrame:
        
        #dropping in-efficient medications > greater than the highest number
        #thermodynamic hard-limit of 10,000
        
        dataframe = self.InChI_based_duplicate_res()
        safe_relations = dataframe["standard_relation"] == "="
        valid_negatives = dataframe["standard_value"] <= 10000
        
        dataframe = dataframe[safe_relations & valid_negatives].copy()
        
        clean_filename = self.get_clean_filename()
        output_path = self.path.parent / f"{clean_filename}_cleaned.csv"
        
        dataframe = dataframe.drop(columns = ["pchembl_value", "potential_duplicate", "relation"])
        dataframe.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved successfully: {output_path}")
        
        return output_path
    

class DataEng:
    
    def __init__(self, path:Path):
        self.path = Path(path) if isinstance(path, str) else path
        self.df = None
    
    @staticmethod
    def morgan_fingerprinting_s_method(smiles_string: str, radius: int = 2, nBits: int = 2) -> np.ndarray:       
        
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return np.zeros(nBits)
        
        finger_printing = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return np.array(finger_printing)
    
    @staticmethod
    def mscl(smiles):
        
        if pd.isna(smiles):
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        return scaffold_smiles
        
        
    def morgan_fingerprinting(self):
        
        self.df = pd.read_csv(self.path)
        self.df['morgan_fp'] = self.df["cleaned_smiles"].apply(
            lambda x: self.morgan_fingerprinting_s_method(x, radius=2, nBits=1024)
        )
        
        X_matrix = np.vstack(self.df["morgan_fp"].values)
        return X_matrix
        
    def scaffold(self, y_vector = None):
        
        X_matrix = self.morgan_fingerprinting()
        
        df = pd.read_csv(self.path)
        df["scaffold"] = df["cleaned_smiles"].apply(DataEng.mscl)
        
        #now giving meaning using InChIkey
        
        df["Scaffold_InChI"] = df["scaffold"].apply(DataCleaning.InChIKeyConversion)
        
        groups = df.groupby("Scaffold_InChI").groups
        
        sorted_keys = sorted(groups.keys(), key = lambda k : len(groups[k]), reverse = True) #ascending or descenfing comes fromthe lamda here actually and we reverse fo rbg uckets to be on the top
        
        target_train_size = int(0.8 * len(df))
        train_indices = []
        test_indices = []
        
        for scaffold in sorted_keys:
            
            row_numbers = groups[scaffold]
            
            if len(train_indices) < target_train_size:
                train_indices.extend(row_numbers)
            
            else:
                test_indices.extend(row_numbers)
                
        X_train = X_matrix[train_indices]
        y_train = y_vector[train_indices]
        
        X_test = X_matrix[test_indices]
        y_test = y_vector[test_indices]
        
        return X_train, y_train, X_test, y_test
              
    
    def variance_thresholding(self):
        
        X_train, y_train, X_test, y_test = self.scaffold()
        
        thresholder = VarianceThreshold(threshold=0.0475) #reasoning in PDF in a more cleaner format
        
        X_train_new = thresholder.fit_transform(X_train)
        X_test_new = thresholder.fit_transform(X_test)
        
        updated_masks = thresholder.get_support()
        
        return X_train_new, X_test_new, updated_masks
        
        
    def correlation_handling(self, threshold = 0.90):
        
        X_train, X_test, updated_masks = self.variance_thresholding()
        #drops one of them to prevent *Impoortance Dilution* in Random-Forests
        
        df_train = pd.DataFrame(X_train)
        df_test = pd.DataFrame(X_test)
        
        #first check for mirror columns to eliminate them
        
        corr_matrix = df_train.corr().abs()
        
        #corr-matrix -- mirror (Double deletion prevention - eliminate the lower matrix)
        
        
        raw_upper_cols = np.triu(corr_matrix, k=1)
        upper_triangle = pd.DataFrame(raw_upper_cols, columns=corr_matrix.columns)
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        X_train_clean = df_train.drop(columns=to_drop).values
        X_test_clean = df_test.drop(columns=to_drop).values
        
        return X_train_clean, X_test_clean
    
    
    def modelledReduction(self, y_train):
    
        X_train, X_test = self.correlation_handling()
        rf = RandomForestRegressor(n_estimators=100, random_state = 50, n_jobs=-1 )
        filterer = SelectFromModel(rf, threshold="mean")
        
        filterer.fit(X_train, y_train)
        
        X_train_raw = filterer.transform(X_train)
        X_test_raw = filterer.transform(X_test)
        
        boolean_array = filterer.get_support()
        surviving_names = X_train.columns[boolean_array]
        
        X_train_final = pd.DataFrame(X_train_raw, columns = surviving_names, index = X_train.index) #here use stencil analogy to row level data manipulation 
        X_test_final = pd.DataFrame(X_test_raw, columns = surviving_names, index = X_test.index)
        
        return X_train_final, X_test_final
    
    def column_addition(self, y_train):
        
        # we have to add back in such a way that the scaffold data does not inherently affect the dimensions here, but here it does not matter 
        
        '''
        My initial thoughts - Even in row concatneation even with scaffolds this would not actually matter because 
        we are some how gonna exterminate the rows that did not survive entirely essentially deleting a dimension. So we can use masking to 
        make the boolean match and work (stencil) for matching dimensions, else use ~ signs or .iloc with truth labels whcih essentially by itself is masking
        
        '''
        d1, d2 = self.modelledReduction(y_train)
        joinee = pd.concat([d1, d2], axis = 0)
        indexes_to_keep = joinee.index.intersection(self.df.index)
        joinee = joinee.loc[indexes_to_keep]
        scaffold_df = self.df.drop(columns=d1.columns, errors='ignore')
        df = pd.concat([scaffold_df, joinee], axis = 1)
        # here we dont want loc based index shuffle since we did not do random shuffling we just did scaffolding
        return df 
    
    
    def encoding(self):
        
        encoder = LabelEncoder()
        self.df["assay_type"] = encoder.fit_transform(self.df["assay_type"])
        
        return self.df 
    
    
    
    
class Model:
    
    def __init__(self):
        pass

class Evaluation:
    
    def __init__(self):
        pass
    
class Explainability:
    
    def __init__(self):
        pass
    
class Representation:
    
    def __init__(self):
        pass
    
class modelserving:
    
    def __init__(self):
        pass
    


if __name__ == "__main__":
    
    ROOT_FOLDER = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()

    target_input = input("Enter the target name (e.g., acetylcholinesterase): ").strip()
    target_results = data_retrieval_desc(target_name=target_input)

    target_idx = int(input("\nEnter the index of the target you want to select: "))
    selected_meta = select_target(target_idx, target_results)

    engine = DuckDBEngine(
        connection=duckdb.connect(),
        targeted=selected_meta,
        root_folder=ROOT_FOLDER
    )
    csv_path, tbl_name = engine.create_and_load_csv()
    engine.create_database(csv_filepath=csv_path, table_name=tbl_name)

    preview_df = engine.query_check(table_name=tbl_name)
    print(preview_df)
    
    dc = DataCleaning(path=csv_path)
    clean_path_output = dc.null_and_columnhandler(path=csv_path)
    #dc.InChIstandardization()
    dc.relationalvalue()