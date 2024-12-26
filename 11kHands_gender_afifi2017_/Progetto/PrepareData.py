import pandas as pd
import numpy as np

def prepare_data(num_exp: int, num_train: int, num_test: int):
    # Load the data from csv metadata file
    df = pd.read_csv('.\\11kHands_gender_afifi2017_\Progetto\HandInfo.csv')
    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    
    # Populate the data structure
    for indExp in range(num_exp):
        print(f"Exp{indExp}")
        data_structure[indExp] = {}
        df['check'] = False
        data_structure[indExp]['train'], df = prepare_data_train(num_train= num_train, df = df)
        data_structure[indExp]['test']= prepare_data_test(num_test= num_test, df=df)  
    return data_structure

def prepare_data_train(num_train: int,  df: pd.DataFrame ):
    result_dict = {
                "labels": [],
                "images": []
            }
    
    gender = ['male',  'female']

    print("Training")

    for gend in gender:
        # Extract the person id without accessories
        person_id_list = df.loc[(df['gender'] == gend), 'id'].unique()
        for _ in range(num_train):
            for i in range(0, len(person_id_list)):
                # Extract a person id
                person_id = np.random.choice(person_id_list)

                '''
                    Escludiamo le persone che non hanno più immagini di palmi e dorsi da estrarre
                '''
                if (len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0)]) == 0 or len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0)]) == 0
                        ) or (
                    df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0), 'check'].all()): 
                  
                    person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])
                    continue 
                else:
                    break
           
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
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            
            # Set the check to 1
            '''
            Con il campo check si indica che un'immagine è già stata  presa e quindi non potrà essere ripescata
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True

            result_dict["images"].append([palmar_img, dorsal_img])

    return result_dict, df

def prepare_data_test(num_test: int, df: pd.DataFrame):
    result_dict = {
        "labels": [],
        "images": []
    } 
    
    male_female_list = ['male', 'female']

    print("Testing")

    for gender in male_female_list:
        person_id_list = df.loc[(df['gender'] == gender), 'id'].unique()

        for person_id in person_id_list:
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

        for _ in range(num_test):
            person_id = np.random.choice(person_id_list)
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
            Con il campo check si indica che un'immagine è già stata  presa e quindi non potrà essere ripescata
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True
            

            result_dict["images"].append([palmar_img, dorsal_img])

            '''
                Escludiamo le persone che non hanno più immagini di palmi e dorsi da estrarre
            '''
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

    return result_dict