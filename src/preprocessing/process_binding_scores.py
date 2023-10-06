"""
Convert binding_affinities.csv => JSON List in FS-MOL format.
"""
import math
import os
import sys
import csv
import jsonlines
import random

from dpu_utils.utils import RichPath

from rdkit.Chem import (
    MolFromSmiles,
)

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from preprocessing.featurisers.molgraph_utils import *


def smiles_list_to_processed_files(smiles_list, target, save_path=None):
    """
    Convert a smiles list => serialized list of json lines.
    """
    metadata_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "utils/helper_files/")
    metapath = RichPath.create(metadata_pth)
    path = metapath.join("metadata.pkl.gz")
    metadata = path.read_by_file_suffix()
    atom_feature_extractors = metadata["feature_extractors"]

    rtn = {target: []}
    for smiles in smiles_list:
        rdkit_mol = MolFromSmiles(smiles)
        g = molecule_to_graph(rdkit_mol, atom_feature_extractors)
        rtn[target].append({
            'SMILES': smiles,
            'graph': g,
        })
    serialize(rtn, save_path, 'test')


def serialize(rtn, save_path, dataset='train'):
    """Serializes rtn dictionary to jsonl file stored on disk.

    Note: Must use .gz ending to be compatible with previous FS-Mol code.
    """
    path = f'{save_path}/{dataset}'
    for key in rtn:
        with jsonlines.open(f'{path}/{key}.jsonl.gz', mode='w') as writer:
            writer.write_all(rtn[key])


def csv_to_processed_files(save_path=None, binding_scores_path='binding_scores/docking_scores.csv'):
    """
    Convert binding_affinities.csv => jsonl file format used in FS-Mol.
    """
    metadata_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "utils/helper_files/")
    metapath = RichPath.create(metadata_pth)
    path = metapath.join("metadata.pkl.gz")
    metadata = path.read_by_file_suffix()
    atom_feature_extractors = metadata["feature_extractors"]

    with open(binding_scores_path) as f:
        csv_data = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]

    rtn = {}
    for row in csv_data:
        if row['protein'] not in rtn:
            rtn[row['protein']] = []
        # Skip this molecule if we don't have a score for it...
        if row['score'] is '':
            continue
        smiles = row['SMILES']
        rdkit_mol = MolFromSmiles(smiles)
        g = molecule_to_graph(rdkit_mol, atom_feature_extractors)
        rtn[row['protein']].append({
            'SMILES': smiles,
            'graph': g,
            'RegressionProperty': float(row['score'])
        })
    random.seed(0)
    train_data = {}
    valid_data = {}
    for target in rtn:
        num_ligands = len(rtn[target])
        valid_split = int(num_ligands * 0.8)
        random.shuffle(rtn[target])
        train_data[target] = (rtn[target])[:valid_split]
        valid_data[target] = (rtn[target])[valid_split:]

    serialize(train_data, save_path, 'train')
    serialize(valid_data, save_path, 'valid')

    print(f'Finished featurizing binding affinity scores.')


# TODO(cfifty): Consider multi-processing this if speedup is worth implementation time.
if __name__ == "__main__":
    raise Exception("this isn't working... Run from FS-Mol original directory instead..")
    path = '../glide_csv_raw'
    for file in os.listdir(path):
        input_path = os.path.join(path, file)
        print(f'processing... :{input_path}')
        csv_to_processed_files(input_path, '../large_binding_datasets')
        print(f'finished processing: {path}')
    # csv_to_processed_files('../large_binding_datasets')
