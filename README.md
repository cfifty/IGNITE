# IGNITE

This repository continas the code
for [Implicit Geometry and Interaction Embeddings Improve Few-Shot Molecular Property Prediction](https://arxiv.org/abs/2302.02055)

If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:

```
@article{fifty2023implicit,
  title={Implicit Geometry and Interaction Embeddings Improve Few-Shot Molecular Property Prediction},
  author={Fifty, Christopher and Paggi, Joseph M and Amid, Ehsan and Leskovec, Jure and Dror, Ron},
  journal={arXiv preprint arXiv:2302.02055},
  year={2023}
}
```

This repository reuses much of the code and structure presented in
the [FS-Mol Benchmark](https://github.com/microsoft/FS-Mol) with changes made to support
training a FS-Mol-style model on GLIDE binding affinity scores.

### To download the IGNITE dataset, simply run:

**Note:** This may take a bit of time. The IGNITE dataset is 77 GB.

```
cd misc/download_data
chmod +x ./download_ignite.sh
./download_ignite.sh
```

### To train a model on the simulation dataset, simply run in the ignite/src directory:

```
python multitask_train.py ../ignite_dataset/ \
--task-list-file datasets/ignite_data.json --num_epochs 101 \
--save-dir ignite_models
```
