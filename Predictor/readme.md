# Protein Feature Prediction Script

## Overview

This Python script allows you to predict features from a protein structure file by specifying a PDB (Protein Data Bank) code. It retrieves the corresponding PDB file, processes it, and outputs the predicted features into a tab-separated values (TSV) file.

## Requirements

### Python Libraries
- `argparse`: For command-line argument parsing.
- `pandas`: For handling data in tabular format.
- `Bio.PDB`: For downloading PDB files.
- `torch`: For neural network model handling.
- `sklearn`: For preprocessing steps, such as label encoding and scaling.
- `xgboost`: For handling XGBoost model predictions.
- `numpy`: For numerical computations.

### Models
- **Neural Network (NN)**: Custom neural network implemented using PyTorch.
- **XGBoost**: Model based on XGBoost library.

### Other Requirements
- `calc_features.py`: A script that calculates features from a PDB file and saves them in a `.tsv` file.

## Script Usage

### 1. Download the Model Checkpoints

Before running the script, you need to download the checkpoints for the models you'll be using and place them in the appropriate directory.

- **XGBoost Model Checkpoint**: Place the `.json` or `.model` file in the `predictor/data/models/` directory.

### 2. Running the Script

The script can be used to predict protein-protein interaction features from a PDB file using either a Neural Network model or an XGBoost model. Below are the steps and command-line arguments to use the script effectively.

### 3. Command-Line Arguments

- **`--pdb <PDB_CODE>`**: (Required) The PDB code of the protein for which you want to predict interaction features. The script will automatically download the corresponding PDB file and extract its features.

- **`--model <MODEL_TYPE>`**: (Optional) The model type to use for prediction. Options include:
  - `"xgboost"`: Use the XGBoost model for prediction.
  - If this argument is not provided, the script will default to the Neural Network model.

- **`--checkpoint <CHECKPOINT_PATH>`**: (Optional) The specific path to the model checkpoint file. If not provided, the script will use the default checkpoint stored in the `predictor/models/` directory.


### Example Usage

To predict features for a specific PDB code:

```bash
python features_predictor.py --pdb 1A2B --model xgboost --checkpoint model.json
```
## Output

After running the script, the predicted interaction features will be saved in a TSV (Tab-Separated Values) file. This output file will be generated in the `./data/output/` directory and will follow the naming convention:
`<PDB_CODE>_predicted.tsv`


### Example

If the PDB code provided is `1A2B`, the output file will be named: `./data/output/1A2B_predicted.tsv`






