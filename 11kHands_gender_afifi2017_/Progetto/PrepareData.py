import pandas as pd
import numpy as np
import json

def prepare_data(num_exp: int, num_train: int, num_test: int):
    # Load the data
    df = pd.read_csv('./Progetto/HandInfo.csv')

    #print(len(df_palmar))
    #print(len(df_dorsal))

    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    train_test_list = ['train', 'test']
    palmar_dorsal_list = ['palmar', 'dorsal']
    male_female_list = ['male', 'female']

    # Populate the data structure
    for indExp in range(num_exp):
        data_structure[indExp] = {}

        for train_test in train_test_list:
            data_structure[indExp][train_test] = {}

            for palmar_dorsal in palmar_dorsal_list:
                data_structure[indExp][train_test][palmar_dorsal] = {}

                for male_female in male_female_list:
                        '''
                        Filtriamo in base al side palmo/dorso
                        Filtriamo in base al gender m/f
                        Nel dataset di training esclusiamo le immagini con ostruzioni (accessories) -> per evitare bias
                        Infine si prende il nome dell'immagine 
                        Con .sample andiamo ad estrarre # num_train o num_test elementi dal dataset e con replace=False evita di estrarre doppioni
                        '''
                        if train_test == 'train':
                            data_structure[indExp][train_test][palmar_dorsal][male_female] = df.loc[
                                        (df['aspectOfHand'].str.contains(palmar_dorsal)) &
                                        (df['gender'] == male_female) &
                                        (df['accessories'] == 0), 'imageName'
                                        ].sample(n=num_train, replace=False).to_list()
                        else:
                             data_structure[indExp][train_test][palmar_dorsal][male_female] = df.loc[
                                  (df['aspectOfHand'].str.contains(palmar_dorsal)) &
                                  (df['gender'] == male_female), 'imageName'
                                  ].sample(n=num_test, replace=False).to_list()

        '''
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
        '''
    
    return data_structure
        
#print(prepare_data(num_exp=1, num_train=5, num_test=5))



       