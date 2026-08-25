import pandas as pd 
reader = pd.read_csv("/media/notshadow/d5dd988b-c393-4302-aa45-32bcfc8463c2/WorkFolder/DrugDiscovery-BioInformatics/database/csv/Acetylcholinesterase_Homo_sapiens_cleaned.csv")

reader.loc[reader["data_validity_comment"] == None , "Outside typical range"] = True

print()
print()

print(reader["data_validity_comment"].value_counts(dropna=False))

print()
print()

'''
Dropping technique: 

1. Find the duplicates create into a dataframe and impute with mean when we have different lab / setting 
2. Then drop the duplicates that has no bias 


3. Standardize the units and relationship modelling --
1. Generate pIC50 for all the molecules -- Done
2. Calculate variance for all of them 
4. Ligand efficency - de-dictionary and small reasearch 
5. Final column drop off 
6. Encoding of features

'''
