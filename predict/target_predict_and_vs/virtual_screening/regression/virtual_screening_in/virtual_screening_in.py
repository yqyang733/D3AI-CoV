import sys, os, re
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

def read_drug(file):
    smiles = []
    names = []
    with open(file) as f:
        f1 = f.readlines()
    for line in f1:
        names.append(line.strip().split("\t")[1])
        smiles.append(line.strip().split("\t")[0])
    return smiles, names

def transmol_linux(in_name,out_name):
    in_suffix=os.path.splitext(in_name)[1][1:]
    out_suffix=os.path.splitext(out_name)[1][1:]
    if out_suffix == "smi":
        command="obabel -i%s %s -ocan -O %s"%(in_suffix,in_name,out_name)
    else:
        command="obabel -i%s %s -o%s -O %s"%(in_suffix,in_name,out_suffix,out_name)
    os.system(command)

def virtual_screen_1_target(smiles, drug_names, target_name, target_fasta, result_folder_ = "./result_test/"):
    #smis = []
    # drug_name = ["drug"]
    # drug_name = np.tile(drug_name, (len(target_name), ))
    # for i in range(len(smiles)):
        # command = "obabel -:'%s' -ocan -O mol_%d.smi"%(smiles[i], i)
    # command = "obabel -:%s -ocan -O mol.smi"%(smiles)
        # os.system(command)
        # with open("mol_%d.smi"%(i)) as f:
            # smi = f.readline().strip()
            # smis.append(smi)
    # print(smiles)
    # smiles = np.tile(smiles, (len(target_name), ))
    target_name = np.tile(target_name, (len(smiles), ))
    target_fasta = np.tile(target_fasta, (len(smiles), ))
    model = models.model_pretrained(path_dir = '/home/yqyang/D3Deep/model/fold_0_model_regression')
    models.virtual_screening(smiles, target_fasta, 
                     model, drug_names = drug_names, target_names = target_name, result_folder = result_folder_,
                     convert_y = False)
    # for i in range(len(smiles)):
        # os.move("./mol_%d.smi"%(i))

def virtual_screen_all_target(smiles, drug_names, target_names, target_fastas):
    for pro in range(len(target_names)):
        virtual_screen_1_target(smiles, drug_names, target_names[pro], target_fastas[pro], result_folder_ = "./result_" + target_names[pro] + "/")

def main():
    molfile_all=str(sys.argv[1])
    mol_name = os.path.splitext(molfile_all)[0]
    transmol_linux(molfile_all, mol_name + ".smi")
    smiles, drug_names = read_drug(mol_name + ".smi")
    # print(mols)
    target_names, target_fastas = read_target()
    # virtual_screen_1_target(mols, target_name, target_fasta)
    virtual_screen_all_target(smiles, drug_names, target_names, target_fastas)

if __name__=="__main__":
    main() 
