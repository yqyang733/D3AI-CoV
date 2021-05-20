import sys, os
import numpy as np
import DeepPurpose.DTI as models

# def read_target():
#     path = ""
#     tar_name = []
#     tar_fasta = []
#     with open("protein_name-fasta.txt") as f:
#         f1 = f.readlines()
#     for line in f1:
#         line = line.strip().split(",")
#         tar_name.append(line[0])
#         tar_fasta.append(line[1])
#     return tar_name, tar_fasta

def read_drug(file):
    smiles = []
    with open(file) as f:
        f1 = f.readlines()
    for line in f1:
        smiles.append(line.strip())
    return smiles

# def virtual_screen_1_target(smiles, target_name, target_fasta, result_folder_ = "./result_test/"):
def virtual_screen_1_target(smiles, target):
    # smis = []
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
    target_name = ["Target"]
    target_name = np.tile(target_name, (len(smiles), ))
    target_fasta = np.tile(target, (len(smiles), ))
    model = models.model_pretrained(path_dir = '/home/databank/D3Deep/model/fold_4_model_regression')
    models.virtual_screening(smiles, target_fasta, 
                     model, drug_names = smiles, target_names = target_name, result_folder = "./result_4/",
                     convert_y = False)
    # for i in range(len(smiles)):
        # os.move("./mol_%d.smi"%(i))

# def virtual_screen_all_target(mols, target_names, target_fastas):
#     for pro in range(len(target_names)):
#         virtual_screen_1_target(mols, target_names[pro], target_fastas[pro], result_folder_ = "./result_" + target_names[pro] + "/")

def main():
    file = sys.argv[1]
    target = sys.argv[2]
    mols = read_drug(file)
    # print(mols)
    # target_names, target_fastas = read_target()
    # virtual_screen_1_target(mols, target_name, target_fasta)
    virtual_screen_1_target(mols, target)

if __name__=="__main__":
    main() 
