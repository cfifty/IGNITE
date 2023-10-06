import os
from schrodinger.structure import StructureReader
import pandas as pd


def _valid_idx_to_smiles():
    rtn = {}
    path = '/scratch/groups/rondror/jpaggi/docking_score_training/ligands/valid/'
    for file in os.listdir(path):
        if file.endswith('.smi'):
            with open(path + file, 'r') as f:
                for line in f.readlines():
                    (smile, valid_idx) = line.split(' ')
                    rtn[valid_idx.strip()] = smile.strip()
    return rtn


def _process_maegz(pv, protein, valid_idx_to_smiles):
    scores = {}
    try:
        with StructureReader(pv) as sts:
            next(sts)  # protein structure is first entry.
            for st in sts:
                key = (protein, st.title)
                score = st.property['r_i_docking_score']
                if key not in scores:
                    scores[key] = score
        scores = [[protein, ligand, valid_idx_to_smiles[ligand], score] for (protein, ligand), score in scores.items()]
        scores = pd.DataFrame(scores, columns=['protein', 'ligand', 'smiles', 'score'])
        scores = scores.set_index(['protein', 'ligand']).sort_index()
    except:
        scores = [[protein, ligand, valid_idx_to_smiles[ligand], score] for (protein, ligand), score in scores.items()]
        scores = pd.DataFrame(scores, columns=['protein', 'ligand', 'smiles', 'score'])
        scores = scores.set_index(['protein', 'ligand']).sort_index()
        print(f'could not process {protein} in path {pv}')

    return scores


def _process_docking_scores(start_count, delta, valid_idx_to_smiles):
    paths = ['/scratch/groups/rondror/jpaggi/docking_score_training/docking']
    dfs = []
    for idx, target in enumerate(os.listdir(''.join(paths + ['/']))):
        # Skip until we get to the correct file.
        if idx < start_count:
            continue
        # if we exceed the limit...
        if idx >= start_count + delta:
            df = pd.concat(dfs, axis=0)
            df.to_csv(f'/scratch/groups/rondror/fifty/full_binding_affinities/{start_count}_{start_count + delta}.csv')
            print(f'Finished: {start_count}')
            return

        paths.append(target)
        for partition in os.listdir('/'.join(paths)):
            paths.append(partition)
            for file in os.listdir('/'.join(paths)):
                if file.endswith('.maegz'):
                    pv = file
                    protein = pv.split('-to-')[-1].split('_pv')[0]
                    paths.append(pv)
                    dfs.append(_process_maegz('/'.join(paths), protein, valid_idx_to_smiles))
                    paths.pop()
            paths.pop()
        paths.pop()
    # In case we reach the end...
    df = pd.concat(dfs, axis=0)
    df.to_csv(f'/scratch/groups/rondror/fifty/full_binding_affinities/{start_count}_{start_count + delta}.csv')
    print(f'Finished.')
    return


# TODO(cfifty): Consider multi-processing this if speedup is worth implementation time.
if __name__ == "__main__":
    # 1932 total files...
    valid_id_to_smiles = _valid_idx_to_smiles()
    d = 20
    # for i in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420,
    #           440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840,
    #           860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220,
    #           1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560,
    #           1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900,
    #           1920]:
    for i in [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420,
              440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840,
              860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220,
              1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560,
              1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900,
              1920]:
        _process_docking_scores(start_count=i, delta=d, valid_idx_to_smiles=valid_id_to_smiles)
