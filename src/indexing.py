import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import faiss
from sentence_transformers import SentenceTransformer
from src.config_loader import CFG

load_dotenv()

def load_song_text(csv_path: str):
	df = pd.read_csv(csv_path)				# Faster than CSVLoader in langchain
	df = df.dropna(subset = ['song_text'], ignore_index=True)
	song_texts_list = df['song_text'].to_list()

	return song_texts_list, df

def generate_embeddings(texts, CFG, device):
	
	model = SentenceTransformer(CFG['indexing']['model_name'], device=device, model_kwargs={"token": os.getenv("HF_KEY")})
	embeddings = model.encode(
		texts,
		batch_size = CFG['indexing']['batch_size'],
		show_progress_bar = True,
		convert_to_numpy=True 			# FAISS only takes numpy array as input, can't convert to tensors
	)
	os.makedirs(os.path.dirname(CFG['indexing']['embeddings_path']), exist_ok=True)
	np.save(CFG['indexing']['embeddings_path'], embeddings)
	return embeddings

def faiss_index(embeddings, CFG):
	faiss.normalize_L2(embeddings)
	index = faiss.IndexFlatIP(CFG['indexing']['embedding_dim'])
	index.add(embeddings)
	faiss.write_index(index, CFG['indexing']['faiss_index_path'])
	return index

def save_metadata(df, CFG):
	df.to_parquet(CFG['data']['metadata_path'])

device = "cuda" if torch.cuda.is_available() else "cpu" 
 
# Load_song_text
texts, df = load_song_text(CFG['data']['processed_path'])

# generate embeddings
embeddings = generate_embeddings(texts, CFG, device)
# print(embeddings.shape)
# print(embeddings.dtype)

# Vector store creation and storage:
# load embeddings from memory (prevent in-place normalization in faiss_index() of og embeddings)
embeddings_load = np.load(CFG['indexing']['embeddings_path'])
embs = embeddings_load.copy()
index = faiss_index(embs, CFG)
# print(index.ntotal)

# saving df for metadata 
save_metadata(df, CFG)
meta = pd.read_parquet(CFG['data']['metadata_path'])
# print(meta.shape)
# print(meta.columns.tolist())