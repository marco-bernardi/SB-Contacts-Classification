# SB-Contacts-Classification

This repository contains the project for the Structural Bioinformatics course of the Master's Degree in Computer Science at the University of Padua (UniPD). The project is focused on the classification of protein contacts using various computational techniques.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Authors](#authors)

## Introduction
The SB-Contacts-Classification project aims to classify contacts within protein structures. This project leverages machine learning algorithms and bioinformatics tools to accurately identify and categorize protein contacts, which are crucial for understanding protein functions and interactions.

## Repository Structure
The repository is organized into the following directories:

- **Predictor**: Contains the source code and scripts used for protein contact classification.
- **FromDataToPrediction.ipynb**: Contains the test and training loops.
    - Follow the comments in the notebook for guidance.
    - Note that to load the already computed SMOTE, you need to have the files `X_bal.npy` and `y_bal.npy` in the same directory as the notebook.
- **Report**: Contains the detailed report of the project, including methodologies, results, and discussions.

## Usage
To use the software, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/SB-Contacts-Classification.git
    cd SB-Contacts-Classification
    ```

2. Navigate to the `Predictor` directory:
    ```sh
    cd Predictor
    ```

3. Follow the folder `readme.md`

## Authors
This project is developed by the following authors:

- Andrea Auletta (2107158): andrea.auletta@studenti.unipd.it
- Marco Bernardi (2107781): marco.bernardi.11@studenti.unipd.it
- Niccolò Zenaro (2125609): niccolo.zenaro@studenti.unipd.it

