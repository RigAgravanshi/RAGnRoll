import os
import faiss
import time
import pandas as pd
from dotenv import load_dotenv
from src.config_loader import CFG
from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()
start = time.time()
index = faiss.read_index(CFG['indexing']['faiss_index_path'])
metadata = pd.read_parquet(CFG['data']['metadata_path'])
model = SentenceTransformer(CFG['indexing']['model_name'], model_kwargs={"token": os.getenv("HF_KEY")})
# embedding_model = HuggingFaceBgeEmbeddings(
#     model_name=CFG['indexing']['model_name'],
#     query_instruction="Represent this sentence for searching relevant passages: ",
#     encode_kwargs={"normalize_embeddings": True})
# vectorstore = FAISS.load_local(							# works only if faiss_index were created initially via langchain. would do it from start
#     CFG['indexing']['faiss_index_path'],
#     embeddings=embedding_model,
#     allow_dangerous_deserialization=True)

def retrieve(prompt:str, k:int = 200):
	# BGE model requires prefix to signal it to work with the prompt
	prefixed = "Represent this sentence for searching relevant passages: " + prompt

	encoded_prompt = model.encode(prefixed)		      	# (384, )
	encoded_prompt = encoded_prompt.reshape(1, -1) 		# (1,384)
	faiss.normalize_L2(encoded_prompt)
	distances, indices = index.search(encoded_prompt, k)

	result_df = metadata.iloc[indices[0]].copy()
	result_df['similarity_score'] = distances[0]
	return result_df
	# results = vectorstore.similarity_search_with_score(prompt, k=k) # results = [(Document, score), (Document, score), ...]
	# rows = []
	# for doc, score in results:
	# 	row = doc.metadata
	# 	row['similarity_score'] = score
	# 	rows.append(row)
	# return pd.DataFrame(rows)

test_prompt = "A classy, slow Italian playlist with mafia-vibes for deep thinking"
results = retrieve(test_prompt, k=50)
print(results[['track_name', 'artist_name', 'genre', 'energy', 'instrumentalness', 'valence', 'similarity_score']])
print(f"Time taken: {time.time()-start}")