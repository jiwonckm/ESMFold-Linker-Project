import os
import torch
import esm
import einops
import openfold
import tree
import ml_collections
from Bio import SeqIO, PDB
import biotite.structure.io as bsio
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBList
from Bio.SVDSuperimposer import SVDSuperimposer
import biotite.structure.io as bsio
import itertools
import gc
import torch

gc.collect()
AA_NAME_MAP = {
  'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
  'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
  'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'TER':'*',
  'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M','XAA':'X'
}
x = pd.read_csv("datasets/antigen_summary.tsv", sep='\t')

def load_pdb(pdb_id, chain_id):
    pdbl = PDBList()
    pdbp = PDBParser()
    pdbl.retrieve_pdb_file(pdb_id.upper(), file_format="pdb", pdir="./")
    struct = pdbp.get_structure("struct", f"pdb{pdb_id.lower()}.ent")
    model = struct[0]

    sequence = []
    coords = []
    for chain in model:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if residue.resname == "HOH":
                continue
            try:
                coords.append(residue['CA'].get_vector().get_array())
            except:
                raise ValueError("There are missing atoms in this structure, try another!")
            resname = AA_NAME_MAP[residue.resname]
            sequence.append(resname)

    return "".join(sequence), np.array(coords)

def load_pdb_file(file_name):
    pdbp = PDBParser()
    struct = pdbp.get_structure("struct", file_name)
    model = struct[0]

    sequence = []
    coords = []
    for chain in model:
        if chain.id != "A":
            continue
    for residue in chain:
        coords.append(residue['CA'].get_vector().get_array())
        resname = AA_NAME_MAP[residue.resname]
        sequence.append(residue.resname)
        
    return "".join(sequence), np.array(coords)

def compute_rmsd(coords1, coords2):
    """This will align coords2 onto coords1."""
    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()
    rms = sup.get_rms()
    return rms

linker = 'G' * 30
linker_len = 120
permutations = list(itertools.permutations(['H', 'L', 'A']))

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

import warnings
warnings.filterwarnings("ignore")

for pdb_id in os.listdir("sabdab_antigen/sabdab_dataset"):
    rmsd_row = [pdb_id]
    plddt_row = [pdb_id, 'NA']

    heavy_id = x[x['pdb']==pdb_id].iloc[0]['Hchain']
    light_id = x[x['pdb']==pdb_id].iloc[0]['Lchain']
    antigen_id = x[x['pdb']==pdb_id].iloc[0]['antigen_chain'][0]

    try:
        heavy_seq, heavy_coords = load_pdb(pdb_id, heavy_id)
        light_seq, light_coords = load_pdb(pdb_id, light_id)
        antigen_seq, antigen_coords = load_pdb(pdb_id, antigen_id)
        info = {'H': (heavy_seq, heavy_coords), 'L': (light_seq, light_coords), 'A': (antigen_seq, antigen_coords)}
    except:
        continue

    # write down all protein IDs that worked
    with open("datasets/proteins2.txt", "a") as file:
        file.write(pdb_id + ", " + heavy_id + ", " + light_id + ", " + antigen_id + "\n")
        
    with torch.no_grad():
        heavy_output = model.infer_pdb(heavy_seq)
        light_output = model.infer_pdb(light_seq)
        antigen_output = model.infer_pdb(antigen_seq)
    heavy_output_path = os.path.join("result", f"{pdb_id}_heavy.pdb")
    light_output_path = os.path.join("result", f"{pdb_id}_light.pdb")
    antigen_output_path = os.path.join("result", f"{pdb_id}_antigen.pdb")
        
    with open(heavy_output_path, "w") as f:
        f.write(heavy_output)
    with open(light_output_path, "w") as f:
        f.write(light_output)
    with open(antigen_output_path, "w") as f:
        f.write(antigen_output)
        
    heavy_output, light_output = load_pdb_file(heavy_output_path)[1], load_pdb_file(light_output_path)[1]
    antigen_output = load_pdb_file(antigen_output_path)[1]
    pred = np.concatenate((heavy_output, light_output, antigen_output), axis=0)
    rmsd = compute_rmsd(truth, pred)
    rmsd_row.append(rmsd)

    for p in permutations:
        first, second, third = p[0], p[1], p[2]
        seq = info[first][0] + linker + info[second][0] + linker + info[third][0]
        truth = np.concatenate((info[first][1], info[second][1], info[third][1]), axis=0)
        first_len, second_len, third_len = info[first][1].shape[0], info[second][1].shape[0], info[third][1].shape[0]
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model.infer_pdb(seq)
        output_path = os.path.join("antigen_result", f"{pdb_id}_{first}{second}{third}.pdb")
        with open(output_path, "w") as f:
            f.write(output)

        struct = bsio.load_structure(output_path, extra_fields=["b_factor"])
        plddt_row.append(struct.b_factor.mean())

        _, output_coords = load_pdb_file(output_path)

        first_output = output_coords[:first_len]
        second_output = output_coords[first_len+linker_len:first_len+linker_len+second_len]
        third_output = output_coords[-third_len:]

        pred = np.concatenate((first_output, second_output, third_output), axis=0)
        try:
            rmsd = compute_rmsd(pred, truth)
            rmsd_row.append(rmsd)
        except:
            rmsd_row.append('failed')


    rmsd_df.append(rmsd_row)
    plddt_df.append(plddt_row)
    
columns = ['ID','Separate','HLA','HAL','LHA','LAH','AHL','ALH']
rmsd_df = pd.DataFrame(rmsd_df, columns = columns)
plddt_df = pd.DataFrame(plddt_df, columns = columns)

rmsd_df.to_csv("rmsd_antigen_results.csv", index=False)
plddt_df.to_csv("plddt_antigen_results.csv", index=False)