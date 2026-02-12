# Motivational Salience Index (MSI) Modeling adapted from M. Pittet

This repository contains the machine learning preliminary analysis pipeline adapted from the original code of Marie Pittet to fit the use case of a Stimulus-Response compatibility task to decode subjective food preference from implicit behavioral task signatures. The data are derived from Malika et al. (2023), they contain: food item liking (VAS ratings), and behavioral responses to tasks featuring these food items. We compared the performance of reuglarized regression (Elastic Net), a tree-based model (HistGB), and recurrent neural networks (LSTM) in predicting food preferences. 

## Project Overview
The goal of this project is to predict individual food liking (VAS scores) using behavioral metrics derived from the Stimulus-Response compatibility (SRC) tasks. We applied within-person centering to isolate cue-specific reactivity.


## Repository Structure
- `src/`: Python scripts for data extraction, preprocessing, and modeling.
- `data/`: (Local only) Raw and preprocessed datasets.
- `results/`: Performance plots and feature importance tables.
- `requirements.txt`: Python dependencies.

## How to Run
1. **Setup:** `pip install -r requirements.txt`
2. **Preprocess:** Run `01_data_extraction.py` through `03_by_item_imputation_normalization.py`.
3. **Model:** Run `04_ML_modelfitting.py` to replicate the ElasticNet results.
