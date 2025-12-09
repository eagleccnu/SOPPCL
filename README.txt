SOPPCL: Sequence-based Order Parameter Prediction with Contrastive Learning

This repository contains the official implementation of SOPPCL, a sequence-only model for predicting residue-level NMR order parameters (S²), built upon protein language models (ESM-2) and contrastive representation learning.

----------------------------------------------------------------------
Overview
----------------------------------------------------------------------

Protein order parameters (S²) describe internal ps-ns timescale motions and are widely used in NMR dynamics analysis. However:

- Experimental S² measurements remain limited.
- Existing machine-learning predictors rely on structural features.

SOPPCL addresses these limitations by learning dynamic-aware sequence representations through:

1. Variational Encoding  
   Compressing high-dimensional PLM embeddings (1280-dim) into a compact latent representation.

2. Contrastive Learning  
   Enforcing invariance under biologically meaningful perturbations (masking, noise, etc.), enhancing dynamic consistency.

3. BiLSTM Regression  
   Mapping latent representations to residue-level S².

Workflow:
Protein Sequence → ESM-2 Embedding + HMM profiles  → VAE Encoder → Contrastive Learning → BiLSTM → S² Prediction

----------------------------------------------------------------------
Repository Structure
----------------------------------------------------------------------

modules.py              - Core modules: VAE, Augmentation, BiLSTM
contrastive_module.py   - Pre-train contrastive learning
regression_module.py    - Train regression model for S²
predict.py              - Evaluate on independent test proteins
utils.py                - PDB/BMRB parsing utilities
README.txt

----------------------------------------------------------------------
Required packages
----------------------------------------------------------------------

- torch - 2.5.1
- numpy - 1.26.4
- dill  - 0.3.8
- scikit-learn  - 1.5.1
- scipy  - 1.12.0
- pandas - 2.2.2
- matplotlib  - 3.9.2
- Bio  - 1.78
- pynmrstar  - 3.3.4

----------------------------------------------------------------------
Data Preparation
----------------------------------------------------------------------

1. Training data (.dat files):
   - ESM-2 embeddings
   - HMM profiles
   - S²: 2755 pseudo-labels by S-OPPE + 26 experimental labels

2. Test set:
   - ./pdb_test/{pdb}.pdb
   - ./nmrstar/{bmrb_id}.str
   - ./esm_hmm_10/{pdb}.dat

Default test proteins:
1PD7, 1WRS, 1WRT, 1Z9B, 2JWT, 2L6B, 2LUO, 2M3O, 2XDI, 4AAI

----------------------------------------------------------------------
Training
----------------------------------------------------------------------

1. Pre-train VAE encoder with contrastive learning:
python contrastive_module.py

Output:
contrastive_model.pth

2. Train BiLSTM regression model:
python regression_module.py

Output:
regression_model.pth

----------------------------------------------------------------------
Testing
----------------------------------------------------------------------

Evaluate on independent proteins:
python predict.py

A results.csv file is generated containing:
PDB | pre | real | PCC | SCC | MAE | RMSE


