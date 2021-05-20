import sys, os
import numpy as np
import DeepPurpose.DTI as models

def read_target():
    path = ""
    tar_name = []
    tar_fasta = []
    with open("protein_name-fasta.txt") as f:
        f1 = f.readlines()
    for line in f1:
        line = line.strip().split(",")
        tar_name.append(line[0])
        tar_fasta.append(line[1])
    return tar_name, tar_fasta

def transmol_linux(in_name,out_name):
    in_suffix=os.path.splitext(in_name)[1][1:]
    out_suffix=os.path.splitext(out_name)[1][1:]
    if out_suffix == "smi":
        command="obabel -i%s %s -ocan -O %s"%(in_suffix,in_name,out_name)
    else:
        command="obabel -i%s %s -o%s -O %s"%(in_suffix,in_name,out_suffix,out_name)
    os.system(command)

def target_predict(smiles, mol_name, target_name, target_fasta):
    drug_name = [mol_name]
    drug_name = np.tile(drug_name, (len(target_name), ))
    #command = "obabel -:'%s' -ocan -O mol.smi"%(smiles)
    #os.system(command)
    #with open("mol.smi") as f:
    #    smiles = f.readline().strip()
    smiles = [smiles]
    smiles = np.tile(smiles, (len(target_name), ))
    model = models.model_pretrained(path_dir = '/home/yqyang/D3Deep/model/fold_7_model_logistic')
    models.virtual_screening(smiles, target_fasta, 
                     model, drug_names = drug_name, target_names = target_name, result_folder = "./result_7/",
                     convert_y = False)

def main():
    mol_file = str(sys.argv[1])
    mol_name = os.path.splitext(mol_file)[0]
    transmol_linux(mol_file, mol_name + ".smi")
    with open(mol_name + ".smi") as f:
        smiles = f.readline().strip().split("\t")[0]
    target_name, target_fasta = read_target()
    target_predict(smiles, mol_name, target_name, target_fasta)

if __name__=="__main__":
    main() 
