from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Optional

from chembl_webresource_client.new_client import new_client
import duckdb
import pandas as pd
from rdkit import Chem 
from rdkit.Chem.SaltRemover import SaltRemover


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

        print("\nBuilding DuckDB tables and persistent storage...")

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
    def InChIConversion(mol):
        
        if pd.isna():
            return None
        
        mol = Chem.MolFromSmiles(mol)
        
        if mol in None:
            return None
        
        InChI = Chem.MolToInchi(mol)
        
        return InChI
    
    @staticmethod
    def InChIKeyConversion(mol):
        
        if pd.isna(mol):
            return None 
        
        mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            return None 
        
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi)
        
        return inchikey
        
    
    def null_and_columnhandler(self, path: Path) -> Path:
        
        dataset_clean = self.loading_csv(path)
        
        dataset_clean = dataset_clean.dropna(axis=1, how="all")
        
        cols_to_drop = ["qudt_units", "uo_units", "toid", "document_chembl_id", "_journal", "_year", 
                        "assay_descriptions", "activity_comment", "upper_value", 
                        "molecule_pref_name"]
        
        existing_cols_to_drop = [col for col in cols_to_drop if col in dataset_clean.columns]
        
        if existing_cols_to_drop:
            dataset_clean = dataset_clean.drop(columns=existing_cols_to_drop)
            print(f"Dropped unneeded columns: {existing_cols_to_drop}")
        
        # on inspection we found out very sparse missing values in these two columns
        
        targets_data_validity_comment = ["Values appear to be an order of magnitude different from previously reported, so units may be incorrect", 
                                         "Potential transcription error"]
        
        targets_data_validity_desc = ["Values for this activity type are unusually large/small, so may not be accurate", 
                                      "Values appear to be an order of magnitude different from previously reported, so units may be incorrect"]
        
        
        mask_1 = dataset_clean["data_validity_comment"].isin(targets_data_validity_comment)
        mask_2 = dataset_clean["data_validity_description"].isin(targets_data_validity_desc)
        
        # AI stub
        combined_bad_rows = mask_1 | mask_2
        rows_to_drop = dataset_clean[combined_bad_rows].index
        dataset_clean = dataset_clean.drop(rows_to_drop, axis = 0)
        #AI stub completed 
        
        #now dropping the left out columns
        dataset_clean = dataset_clean.drop(columns=["data_validity_comment", "data_validity_description"], axis=1)
        
        #my stub
        clean_filename = self.get_clean_filename()
        output_path = path.parent / f"{clean_filename}_cleaned.csv"
        
        dataset_clean.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved successfully: {output_path}")
        
        return output_path

    
    def molecule_standardization(self, cleaned_csv_path : Path = null_and_columnhandler) -> pd.DataFrame:
        
        df = pd.read_csv(cleaned_csv_path)
        df["cleaned_smiles"] = df["canonical_smiles"].apply(DataCleaning.stip_salt)
        
        df.drop("canonical_smiles", axis=1, inplace=True)
        
        #now creating a InChI key 
        df["InChI"] = df["cleaned_smiles"].apply(DataCleaning.InChIConversion)
        df["InChIkey"] = df["cleaned_smiles"].apply(DataCleaning.InChIKeyConversion)
        
        file_name = cleaned_csv_path
        
        df.to_csv(f"{file_name}")
        
        
    def duplicate_resolution(self):
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
    cleaned_csv= dc.null_and_columnhandler(csv_path)
    dc.molecule_standardization(cleaned_csv_path=cleaned_csv)