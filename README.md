# Microstructure Alpha Engine

Predict short-term price movements using limit order book microstructure features.

## Project Structure
microstructure_alpha/ – core code  
data/ – datasets  
notebooks/ – experiments  
models/ – trained models  

## Setup

conda env create -f environment.yml
conda activate lob-alpha

## Run

python microstructure_alpha/models/train.py