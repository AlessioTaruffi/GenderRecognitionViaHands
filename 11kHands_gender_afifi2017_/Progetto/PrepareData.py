import pandas as pd
import numpy as np
import json

def prepare_dataOld(num_exp: int, num_train: int, num_test: int):
    # Load the data
    df = pd.read_csv('/Users/Candita/Desktop/ComputerScience/FDS/GenderRecognitionViaHands/11kHands_gender_afifi2017_/Progetto/HandInfo.csv')

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
    
    return data_structure

'''
Il numero di immagini estratte per il testing e il test è errato -> porta a 100% correttezza :(
'''
def prepare_data(num_exp: int, num_train: int, num_test: int):
    # Load the data from csv metadata file
    df = pd.read_csv('/Users/Candita/Desktop/ComputerScience/FDS/GenderRecognitionViaHands/11kHands_gender_afifi2017_/Progetto/HandInfo.csv')
    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    train_test_list = ['train', 'test']
    set1= set()
    set2 = set()

    
    # Populate the data structure
    for indExp in range(num_exp):
        #print(f"Exp{indExp}")
        data_structure[indExp] = {}
        df['check'] = False
        #print('\tTrain')
        data_structure[indExp]['train'], set1, val_est, df = prepare_data_train(num_train= num_train, df = df)

        #print(f"\n{df.loc[df['check'] == False, 'id'].unique()}\n")
        
        #print('\tTest')
        data_structure[indExp]['test'], set2 = prepare_data_test(num_test= num_test, df=df)  
    return data_structure

def prepare_data_train(num_train: int,  df: pd.DataFrame ):
    result_dict = {
                "labels": [],
                "images": []
            }
    settino = set()
    val_est = []

    #person_id_no_accessories_list = df.loc[df['accessories'] == 0, 'id'].unique()
    gender = ['male',  'female']

    for gend in gender:
        # Extract the person id without accessories
        person_id_no_accessories_list_gender =  df.loc[(df['accessories'] == 0) & (df['gender'] == gend), 'id'].unique()

        #print(f"\t\t{gend} Len persone Iniziale:{len(person_id_no_accessories_list_gender)}")
        for _ in range(num_train):
            '''
            # Il problema di questa cosa è che potresti non arrivare al numero di immagini richieste 
            if len(person_id_no_accessories_list_gender) == 0:
                return result_dict, settino
            '''

            # Extract a person id
            person_id = np.random.choice(person_id_no_accessories_list_gender)
            #print(f"\t\t\tPersona:{person_id}")

            '''
            Filtriamo in base al side palmo/dorso
            Nel dataset di training esclusiamo le immagini con ostruzioni (accessories) -> per evitare bias
            Infine si prende il nome dell'immagine 
            Con .sample andiamo ad estrarre # num_train o num_test elementi dal dataset e con replace=False evita di estrarre doppioni
            '''
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
            '''
            Da tutto il datafram df 
            filtriamo su id di una singola persona
            prendo i palmi o dorsi
            Scegliamo in modo casuale un palmo e una mano
            Con check == True l'immagine viene esclusa perchè già presa 
            '''  
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            
            # Set the check to 1
            '''
            Con il campo check si indica che un'immagine è già stata  presa e quindi non potrà essere ripescata
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True

            #print(df.loc[(df["imageName"] == palmar_img[0]),'check'])
            #print(df.loc[(df["imageName"] == dorsal_img[0]),'check'])

            result_dict["images"].append([palmar_img, dorsal_img])

            '''
                Escludiamo le persone che non hanno più immagini di palmi e dorsi da estrarre
            '''
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                #print(f"\t\t\t\tPersona esclusa:{person_id}")
                person_id_no_accessories_list_gender = np.delete(person_id_no_accessories_list_gender, np.where(person_id_no_accessories_list_gender == person_id)[0])
            
            val_est.append(palmar_img)
            val_est.append(dorsal_img)
            settino.update(palmar_img) 
            settino.update(dorsal_img)  
        #print(f"\t\tLen persone Finale:{len(person_id_no_accessories_list_gender)}")
    return result_dict, settino, val_est, df

def prepare_data_test(num_test: int, df: pd.DataFrame):
    result_dict = {
        "labels": [],
        "images": []
    } 
    settino = set()
    male_female_list = ['male', 'female']

    for gender in male_female_list:

        person_id_list = df.loc[(df['gender'] == gender), 'id'].unique()

        '''
        if gender == 'female':
            print("Numero di femmine: " + str(len(person_id_list)))
        else:            
            print("Numero di maschi: " + str(len(person_id_list)))
        '''

        for person_id in person_id_list:
            
            #print(str( df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))])  + " " + str( df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')),'check'].any() ) )
            #print(str( df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))])  + " " + str( df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].any() ) )
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].any() and df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].any():
                #print(f"\t\t\t\tPersona esclusa TRAINING:{person_id}")
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

        #print(f"\t\t{gender} Len persone Iniziale:{len(person_id_list)}")
        for _ in range(num_test):
            '''
            # Il problema di questa cosa è che potresti non arrivare al numero di immagini richieste 
            if len(person_id_list) == 0:
                return result_dict, settino
            '''

            person_id = np.random.choice(person_id_list)
            #print(f"\t\t\tPersona:{person_id}")
            '''
            Filtriamo in base al side palmo/dorso
            Nel dataset di training esclusiamo le immagini con ostruzioni (accessories) -> per evitare bias
            Infine si prende il nome dell'immagine 
            Con .sample andiamo ad estrarre # num_train o num_test elementi dal dataset e con replace=False evita di estrarre doppioni
            '''
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
            '''
            Da tutto il dataframe df 
            filtriamo su id di una singola persona
            prendo i palmi o dorsi
            Scegliamo in modo casuale un palmo e una mano
            '''

            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            
            '''
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar")),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal")),'imageName'].sample(n=1, replace=False).to_list()
            '''

            '''
            SE VOGLIAMO FILRARE LE IMMAGINI NON PRESE
            Con il campo check si indica che un'immagine è già stata  presa e quindi non potrà essere ripescata
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True
            

            result_dict["images"].append([palmar_img, dorsal_img])

            '''
                SE VOGLIAMO FILRARE LE IMMAGINI NON PRESE
                Escludiamo le persone che non hanno più immagini di palmi e dorsi da estrarre
            '''

            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                #print(f"\t\t\t\tPersona esclusa:{person_id}")
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

            settino.update(palmar_img) 
            settino.update(dorsal_img)

        #print(f"\t\t{gender}Len persone Finale:{len(person_id_list)}")
    return result_dict, settino

prepare_data(10, 100, 50)