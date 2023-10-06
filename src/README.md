This repository is a slimmed-down version of FS-Mol: https://github.com/microsoft/FS-Mol

To train a model on the simulation dataset, simply run:
```
python multitask_train.py ../simulation_dataset/ \
--task-list-file datasets/simulation_data.json --num_epochs 101 \
--save-dir simulation_models
```

### Note: You will have to change datasets/simulation_data.json to include a json of your targets (i.e. the folder names).

You may also test training a model by changing the data_path to ```ignite_src/prototype_simulation_dataset``` like below:
```
python multitask_train.py ignite_src/prototype_simulation_dataset/ \
--task-list-file datasets/simulation_data.json --num_epochs 101 \
--save-dir simulation_models
```