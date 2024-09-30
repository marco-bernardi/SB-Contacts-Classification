import argparse, os
import pandas as pd
from Bio.PDB import PDBList
from model import Predictor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xgboost as xgb

def parse_args():
    parser = argparse.ArgumentParser(description='Predict features from a PDB file')
    parser.add_argument('--pdb', type=str, help='PDB code of the protein', required=True)
    parser.add_argument('--model', type=str, help='Model to use for prediction', required=False)
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint', required=False)

    return parser.parse_args()

def predict_features(pdb_code, model_type=None, checkpoint=None):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb_code, pdir='./data/output/')
    try:
        os.system(f"python3 ./data/script/calc_features.py ./data/output/{pdb_code}.cif -out_dir ./data/output/")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {pdb_code}.cif not found")
        exit()

    labels = {0: "HBOND", 1: "Unclassified", 2: "VDW", 3: "PIPISTACK", 4: "IONIC", 5: "PICATION", 6: "PIHBOND", 7: "SSBOND"}

    try:
        features = pd.read_csv(f'./data/output/{pdb_code}.tsv', sep='\t')
        if model_type == "nn":
            if checkpoint is None:
                model = Predictor()
                print(f"Predicting features using default model")
            else:
                model = Predictor()
                model.load(checkpoint)
                print(f"Predicting features using default model {checkpoint}")
            output = model.predict(dataframe_to_tensor(features))
            features['Interaction'] = output.numpy()
            features['Interaction'] = features['Interaction'].map(labels)
        elif model_type == 'xgboost':
            print(f'Predicting features using XGBoost model')
            bst = xgb.Booster()
            if checkpoint is None:
                print(f'No checkpoint provided, loading default model')
                bst.load_model('./data/models/xgboost_model.json')
            else:
                print(f'Loading model from checkpoint {checkpoint}')
                bst.load_model(checkpoint)
            xgbMatrix = xgb.DMatrix(clean_features(features), enable_categorical=True)
            output = np.argmax(bst.predict(xgbMatrix), axis=1)
            features['Interaction'] = output
            features['Interaction'] = features['Interaction'].map(labels)
            print(f'Predicted features of {pdb_code} saved to ./data/output/{pdb_code}_predicted.tsv')
        else:
            print(f'Predicting features using default model')
            output = model.predict(dataframe_to_tensor(features))
            # append to features dataframe a new column with the predicted output
            features['Interaction'] = output.numpy()
            features['Interaction'] = features['Interaction'].map(labels)   
            # Load model and predict features
            # Save the features dataframe with the predicted output
        
        features.to_csv(f'./data/output/{pdb_code}_predicted.tsv', sep='\t', index=False)

    except FileNotFoundError:
        raise FileNotFoundError(f"Features file of {pdb_code} not found")
        exit() 

def clean_features(df):
    le = LabelEncoder()
    # Explicitly create a copy of the DataFrame slice
    X = df[['s_ss8', 's_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 
           't_ss8', 't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']].copy()
    
    X['s_ss8_encoded'] = le.fit_transform(X['s_ss8'])
    X = X.drop(columns=['s_ss8'])
    X['t_ss8_encoded'] = le.fit_transform(X['t_ss8'])
    X = X.drop(columns=['t_ss8'])
    
    # Fill None value with mean of the column
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    print("Features cleaned")
    return X


def dataframe_to_tensor(df):
    le = LabelEncoder()
    X = df[['s_ss8','s_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_ss8','t_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
    X['s_ss8_encoded'] = le.fit_transform(X['s_ss8'])
    X = X.drop(columns=['s_ss8'])
    X['t_ss8_encoded'] = le.fit_transform(X['t_ss8'])
    X = X.drop(columns=['t_ss8'])
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    # Replace NaN in columns s_rsa, t_rsa with 0
    X['s_rsa'] = X['s_rsa'].fillna(0)
    X['t_rsa'] = X['t_rsa'].fillna(0)    
    minMax = MinMaxScaler()
    X_scaled = minMax.fit_transform(X)
    X = np.array(X_scaled)
    return torch.tensor(X, dtype=torch.float32)

def main():
    args = parse_args()
    pdb_code = args.pdb
    model = args.model
    checkpoint = args.checkpoint
    if pdb_code is None:
        raise ValueError('Please provide a PDB code')
    else:
        predict_features(pdb_code, model, checkpoint)


if __name__ == '__main__':
    main()