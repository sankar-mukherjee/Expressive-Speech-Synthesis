# Expressive Speech Synthesis

An Expressive Speech synthesis system with blizzard 2013 dataset support.

## Requirements

``` shell
pip3 install -r requirements.txt
```

## File structure

- `Hyperparameters.py` --- contain all hyperparameters
- `Data.py` --- load dataset
- `utils.py` --- some util functions for data I/O

## How to train
- Download a blizzard 2013 dataset
- create filelists folder
- run blizzard_preprocess.py which split data into training and validation
- Adjust hyperparameters  in `Hyperparameters.py`}