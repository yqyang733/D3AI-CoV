<h3 align="center">
<p> D3AI-CoV<br></h3>

## News   

## Train   
### Target Predicting   
Model 1: [MPNN-CNN-Logistic](train/Deep/logistic/DeepPurpose_DeepPurpose_outtest/); run python D-MPNN_T-CNN.py ./D3Similarity_data/drug_smiles.txt ./D3Similarity_data/protein_fasta.txt ./D3Similarity_data/drug_protein.txt   

Model 2: [Commom-Space-Predict](train/common_space/D3similarity/); run python main.py    

### Virtual Screening    
Model: [MPNN-CNN-Regression](train/Deep/regression/D-MPNN_T-CNN-regression/); run python D-MPNN_T-CNN-regression.py ./D3Similarity_data/drug_smiles.txt ./D3Similarity_data/protein_fasta.txt ./D3Similarity_data/drug_protein.txt    

## Predict    
### Target Predicting     
Model 1: [MPNN-CNN-Logistic](predict/target_predict_and_vs/target_predict/in/); run python target_predicting_in.py icccc.sdf

Model 2: [Commom-Space-Predict](predict/target_predict_and_vs/target_predict/common_space/); run python predict_1.py icccc.sdf

### Virtual Screening    
Model: [MPNN-CNN-Regression](predict/target_predict_and_vs/virtual_screening/regression/virtual_screening_in/); run python virtual_screening_in.py 41_top100.sdf    
