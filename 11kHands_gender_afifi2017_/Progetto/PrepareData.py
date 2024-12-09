import pandas as pd
import numpy as np
import json

def prepare_data(num_exp: int, num_train: int, num_test: int):
    # Load the data
    df = pd.read_csv('./Progetto/HandInfo.csv')

    # Create palmar and dorsal dataframes
    df_palmar = df[df['aspectOfHand'].str.contains('palmar')]
    df_dorsal = df[df['aspectOfHand'].str.contains('dorsal')]

    #print(len(df_palmar))
    #print(len(df_dorsal))

    # Create a flag column to identify the chosen images
    df_palmar['chosen'] = 0
    df_dorsal['chosen'] = 0

    #print(df_palmar.head())
    #print(df_dorsal.head())

    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}

    # Populate the data structure
    for indExp in range(num_exp):
        data_structure[indExp] = {}
    
        ## Training set
        data_structure[indExp]['train'] = {}

        ### Palmar
        data_structure[indExp]['train']['palmar'] = {}

        #### Get the palmar images' name
        data_structure[indExp]['train']['palmar'].update(df_palmar['imageName'].sample(n=num_train, replace=False))

        ### Dorsal
        data_structure[indExp]['train']['dorsal'] = {}

        #### Get the dorsal images' name
        data_structure[indExp]['train']['dorsal'].update(df_dorsal['imageName'].sample(n=num_train, replace=False))

        ## Test set
        data_structure[indExp]['test'] = {}

        ### Palmar
        data_structure[indExp]['test']['palmar'] = {}

        #### Get the palmar images' name
        data_structure[indExp]['test']['palmar'].update(df_palmar['imageName'].sample(n=num_test, replace=False))

        ### Dorsal
        data_structure[indExp]['test']['dorsal'] = {}
        
        #### Get the dorsal images' name
        data_structure[indExp]['test']['dorsal'].update(df_dorsal['imageName'].sample(n=num_test, replace=False))
    
    return data_structure
        
print(json.dumps(prepare_data(num_exp=1, num_train=5, num_test=5)))



       