# IDOL

## Environment Installation
```cpp
conda create -n idol python=3.8
conda activate idol
pip install -e . 
```

## Prepare dataset
Due to file size limitation, we keep only Dataset A which is smallest. For other datasets, please refer to '''IDOL/tools/gen_data.py'''.

## Run
- go to the scripts folder
```
cd scripts
```
- run the scrips
```
bash train.sh
```
