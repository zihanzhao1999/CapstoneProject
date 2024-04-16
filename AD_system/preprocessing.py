""""
This script further preprocesses the data for AD system.
"""
import argparse
import ast
import pandas as pd
import numpy as np


def convert_list_diagnosis(diagnosis_list):
    if isinstance(diagnosis_list, str):
        try:
            return ast.literal_eval(diagnosis_list)
        except Exception as e: 
            return None 
    else:
        return None
    

def one_hot_encode_diagnosis(df, column_name):
    # ensure that each diagnosis list is correctly formatted
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])

    # flatten the list of all possible diagnoses, taking only the diagnosis name (first item of each sublist)
    all_diagnoses = set(diagnosis[0] for sublist in df[column_name] for diagnosis in sublist if isinstance(diagnosis, list) and len(diagnosis) > 0)
    
    # initialize a dictionary to hold the one-hot encoded data
    one_hot_encoded_data = {diagnosis: [] for diagnosis in all_diagnoses}
    
    # populate the dictionary with 1s and 0s based on diagnosis presence
    for index, row in df.iterrows():
        present_diagnoses = {diagnosis[0] for diagnosis in row[column_name] if isinstance(diagnosis, list) and len(diagnosis) > 0}
        for diagnosis in all_diagnoses:
            one_hot_encoded_data[diagnosis].append(1 if diagnosis in present_diagnoses else 0)
    
    # convert the dictionary to a DataFrame
    one_hot_y = pd.DataFrame(one_hot_encoded_data)
    
    return one_hot_y


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", help="intput path to the patient data file (csv file)", type=str)
    parser.add_argument(
        "--output_path", help="output path", type=str, default="data")
    args = parser.parse_args()
    
    data = pd.read_csv(args.input_path)
    data['DIFFERENTIAL_DIAGNOSIS'] = data['DIFFERENTIAL_DIAGNOSIS'].apply(convert_list_diagnosis)
    data_y = one_hot_encode_diagnosis(data, 'DIFFERENTIAL_DIAGNOSIS')
    data = data.drop(['PATHOLOGY', 'DIFFERENTIAL_DIAGNOSIS', 'INITIAL_EVIDENCE'], axis=1)
    data.astype(np.int8).to_pickle(f'{args.output_path}_x.zst')
    data_y.astype(bool).to_pickle(f'{args.output_path}_y.zst')
    
