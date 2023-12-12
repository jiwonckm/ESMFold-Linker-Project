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
AA_NAME_MAP = {
  'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
  'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
  'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'TER':'*',
  'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M','XAA':'X'
}
x = pd.read_csv("datasets/summary.tsv", sep='\t')

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

def hl(h, l):
    return h+l
def lh(h, l):
    return l+h

linker1 = 'G' * 15
def hl_linker1(h, l):
    return h+linker1+l
def lh_linker1(h, l):
    return l+linker1+h

linker2 = 'G' * 30
def hl_linker2(h, l):
    return h+linker2+l
def lh_linker2(h, l):
    return l+linker2+h

linker3 = 'GGGGS' * 3
def hl_linker3(h,l):
    return h+linker3+l
def lh_linker3(h,l):
    return l+linker3+h

linker4 = 'P' * 15
def hl_linker4(h,l):
    return h+linker4+l
def lh_linker4(h,l):
    return l+linker4+h

# Download the model
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

import warnings
warnings.filterwarnings("ignore")

rmsd_df = []
plddt_df = []

for pdb_id in os.listdir("sabdab_dataset"):
    rmsd_row = [pdb_id]
    plddt_row = [pdb_id, 'NA']
    
    heavy_id = x[x['pdb']==pdb_id].iloc[0]['Hchain']
    light_id = x[x['pdb']==pdb_id].iloc[0]['Lchain']
    
    try:
        heavy_seq, heavy_coords = load_pdb(pdb_id, heavy_id)
        light_seq, light_coords = load_pdb(pdb_id, light_id)
        heavy_len = heavy_coords.shape[0]
        light_len = light_coords.shape[0]
        hl_truth = np.concatenate((heavy_coords, light_coords), axis=0)
        lh_truth = np.concatenate((light_coords, heavy_coords), axis=0)
    except:
        continue
        
    # write down all protein IDs that worked and their heavy/lightchain IDs
    with open("datasets/proteins.txt", "a") as file:
        file.write(pdb_id + ", " + heavy_id + ", " + light_id + "\n")
        
        
    # separate
    with torch.no_grad():
        heavy_output = model.infer_pdb(heavy_seq)
        light_output = model.infer_pdb(light_seq)
    heavy_output_path = os.path.join("result", f"{pdb_id}_heavy.pdb")
    light_output_path = os.path.join("result", f"{pdb_id}_light.pdb")
        
    with open(heavy_output_path, "w") as f:
        f.write(heavy_output)
    with open(light_output_path, "w") as f:
        f.write(light_output)
        
    heavy_output, light_output = load_pdb_file(heavy_output_path)[1], load_pdb_file(light_output_path)[1]
    pred = np.concatenate((heavy_output, light_output), axis=0)
    rmsd = compute_rmsd(truth, pred)
    rmsd_row.append(rmsd)
    
    
    # heavy first
    for i, concat_method in enumerate([hl, hl_linker1, hl_linker2, hl_linker3, hl_linker4]):
        with torch.no_grad():
            output = model.infer_pdb(concat_method(heavy_seq, light_seq))
        output_path = os.path.join("result", f"{pdb_id}_{i}.pdb")
        with open(output_path, "w") as f:
            f.write(output)
        
        # Compute pLDDT
        struct = bsio.load_structure(output_path, extra_fields=["b_factor"])
        plddt_row.append(struct.b_factor.mean())
        
        # Compute RMSD
        _, output_coords = load_pdb_file(output_path)
        
        heavy_output = output_coords[:heavy_len]
        light_output = output_coords[-light_len:]
        
        hl_pred = np.concatenate((heavy_output, light_output), axis=0)
        rmsd = compute_rmsd(hl_pred, hl_truth)
        rmsd_row.append(rmsd)
        
        
    # light first
    for i, concat_method in enumerate([lh, lh_linker1, lh_linker2, lh_linker3, lh_linker4]):
        with torch.no_grad():
            output = model.infer_pdb(concat_method(heavy_seq, light_seq))
        output_path = os.path.join("result", f"{pdb_id}_{i+5}.pdb")
        with open(output_path, "w") as f:
            f.write(output)
            
        # Compute pLDDT
        struct = bsio.load_structure(output_path, extra_fields=["b_factor"])
        plddt_row.append(struct.b_factor.mean())
        
        # Compute RMSD
        _, output_coords = load_pdb_file(output_path)
        
        light_output = output_coords[:light_len]
        heavy_output = output_coords[-heavy_len:]
        
        lh_pred = np.concatenate((light_output, heavy_output), axis=0)
        
        rmsd = compute_rmsd(lh_pred, lh_truth)
        rmsd_row.append(rmsd)
        
        
    rmsd_df.append(rmsd_row)
    plddt_df.append(plddt_row)

    
columns = ['ID','Separate', 'hl','hl_G15','hl_G30','hl_GS15','hl_P15','lh','lh_G15','lh_G30','lh_GS15','lh_P15']
rmsd_df = pd.DataFrame(rmsd_df, columns = columns)
plddt_df = pd.DataFrame(plddt_df, columns = columns)

rmsd_df.to_csv("rmsd_results.csv", index=False)
plddt_df.to_csv("plddt_results.csv", index=False)