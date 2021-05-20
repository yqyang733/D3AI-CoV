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

def target_predict(smiles, fasta):
    drug_name = ["drug"]
    target_name = ["Target"]
    # drug_name = np.tile(drug_name, (len(target_name), ))
    #command = "obabel -:'%s' -ocan -O mol.smi"%(smiles)
    #os.system(command)
    #with open("mol.smi") as f:
    #    smiles = f.readline().strip()
    smiles = [smiles]
    # smiles = np.tile(smiles, (len(target_name), ))
    model = models.model_pretrained(path_dir = '/home/databank/D3Deep/model/fold_7_model_logistic')
    models.virtual_screening(smiles, fasta, 
                     model, drug_names = drug_name, target_names = target_name, result_folder = "./result_7/",
                     convert_y = False)

def main():
    smiles = sys.argv[1]
    fasta = sys.argv[2]
    # target_name, target_fasta = read_target()
    target_predict(smiles, fasta)

if __name__=="__main__":
    main() 
