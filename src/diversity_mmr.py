import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

features = ['energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness', 'danceability']

def _normalize_name(name: str) -> str:  # _ used in func name bcoz: its a pvt. helper, naming convention
	name = name.lower()
	name = re.sub(r'[^a-z0-9 ]', '', name)    
	name = re.sub(r'\b(\w+?)(ing|tion|s|es)\b', r'\1', name)
	name = name.strip()
	return name

def deduplicate_by_name(songs: pd.DataFrame) -> pd.DataFrame:
	songs_df = songs.copy()
	songs_df["norm_name"] = songs_df["track_name"].apply(_normalize_name)
	songs_df = (
		songs_df
		.sort_values(by="final_score", ascending=False)
		.groupby(by="norm_name", as_index=False)
		.first() 
		)
	# groupby().first(): as_index=False + first() keeps all columns, takes first row per group(highest final_score after sort).
	return songs_df.sort_values("final_score", ascending=False).reset_index(drop=True)

def apply_mmr(songs: pd.DataFrame, intent: dict, lambda_param: float = 0.7) -> pd.DataFrame:
    playlist_length = intent.get('playlist_length', 15)
    candidates = songs.reset_index(drop=True)

    selected_indices = []
    remaining_indices = list(candidates.index)
    
    # seed with highest final_score unconditionally
    first_idx = candidates['final_score'].idxmax()
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # iteratively pick best MMR candidate
    while len(selected_indices) < playlist_length and remaining_indices:
        selected_vecs = candidates.loc[selected_indices, features].values
        
        best_idx = None
        best_score = -np.inf
        
        for idx in remaining_indices:
            candidate_vec = candidates.loc[idx, features].values.reshape(1, -1)
            sims = cosine_similarity(candidate_vec, selected_vecs)
            max_sim = sims.max()
            
            mmr_score = lambda_param * candidates.loc[idx, 'final_score'] - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return candidates.loc[selected_indices].reset_index(drop=True)


if __name__ == "__main__":
	from src.ranker import rank_songs
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent
	print("Imports COMPLETED")
	test_prompt = "I want songs for a late-night train journey after a difficult breakup."
	intent = parse_intent(test_prompt)
	print("PARSER.PY DONE")
	query = rebuild_retrieval_query(intent)
	print("QUERY REBUILT DONE")
	results = retrieve(query, k=1500)
	print("RETRIEVAL DONE")
	filtered = filter_songs(results, intent)
	print("FILTERING ALSO DONE!! Now Ranking......")
	ranked = rank_songs(filtered, intent)
	print("Ranking of Kings DONE")
	dedup_df = deduplicate_by_name(ranked)

	print(f"Ranked pool size: {len(ranked)}")
	print(f"After dedup: {len(dedup_df)}")

	final_result = apply_mmr(dedup_df, intent, lambda_param = 0.7)
	
	# print("=== BEFORE MMR ===")
	# print(ranked[['track_name', 'final_score']].to_string())
	print("\n=== AFTER MMR ===")
	print(final_result[['track_name', 'artist_name', 'genre', 'mood_score', 'normalized_similarity']].to_string())
