from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

history_expression = "persistent headaches"
allergies_expression = "pollen, mold and pet dander"

def diff_embeddings(e1, e2):
    dot_product = np.dot(e1, e2)
    
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    
    return 1 - dot_product / (norm_e1 * norm_e2)

def use_embeddings():
    embeddings = []
    inputs = tokenizer([history_expression, allergies_expression], return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings.extend(outputs.last_hidden_state.mean(dim=1).numpy())
    
    return embeddings[0], embeddings[1]

def calculate_reference_embedding(data, embedding_cols):
    
    filtered_data = data[data['hasHeadache'] == 1]

    embeddings = filtered_data[embedding_cols].values

    reference_embedding = embeddings.mean(axis=0)

    return reference_embedding


def transform_data(data):
    history_cols = [col for col in data.columns if col.startswith(f"HISTORY_dim")]
    allergies_cols = [col for col in data.columns if col.startswith(f"ALLERGIES_dim")]

    history_embeddings = data[history_cols].values
    allergies_embeddings = data[allergies_cols].values 
    
    ref_history_embedding = calculate_reference_embedding(data, history_cols)
    ref_allergies_embedding = calculate_reference_embedding(data, allergies_cols)
    
    history_distances = [diff_embeddings(emb, ref_history_embedding) for emb in history_embeddings]
    allergies_distances = [diff_embeddings(emb, ref_allergies_embedding) for emb in allergies_embeddings]
    
    data = data.drop(history_cols + allergies_cols, axis=1)
    
    data['HISTORY_distance'] = history_distances
    data['ALLERGIES_distance'] = allergies_distances
    
    return data
    



if __name__ == '__main__':
    
    data = pd.read_csv('data_processed.csv')
    
    data = transform_data(data)
    
    data.to_csv('data_processed_transformed.csv', index=False)