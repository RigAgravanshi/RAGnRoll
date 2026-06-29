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
	slim_path = CFG['data'].get('metadata_slim_path')
	if not slim_path:
		return
	slim_cols = [
		'track_name', 'artist_name', 'genre', 'popularity',
		'energy', 'valence', 'acousticness', 'instrumentalness',
		'speechiness', 'danceability',
	]
	slim = df[slim_cols].copy()
	slim['genre'] = slim['genre'].astype('category')
	slim['popularity'] = pd.to_numeric(slim['popularity'], downcast='integer')
	for col in slim_cols[4:]:
		slim[col] = slim[col].astype('float32')
	os.makedirs(os.path.dirname(slim_path), exist_ok=True)
	slim.to_parquet(slim_path, compression='zstd', index=False)


def build_slim_metadata():
	"""Build track_metadata_slim.parquet from existing full metadata only."""
	meta_path = CFG['data']['metadata_path']
	if not os.path.isfile(meta_path):
		raise FileNotFoundError(f"Full metadata not found: {meta_path}")
	df = pd.read_parquet(meta_path)
	save_metadata(df, CFG)
	print(f"Wrote slim metadata to {CFG['data']['metadata_slim_path']}")


def run_full_index():
	texts, df = load_song_text(CFG['data']['processed_path'])
	generate_embeddings(texts, CFG, device)
	embeddings_load = np.load(CFG['indexing']['embeddings_path'])
	embs = embeddings_load.copy()
	faiss_index(embs, CFG)
	save_metadata(df, CFG)


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == "--slim-only":
		build_slim_metadata()
	else:
		run_full_index()