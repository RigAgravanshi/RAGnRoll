import pandas as pd
from src.retriever import features

def evaluate_playlist(df: pd.DataFrame, intent: dict) -> dict :
	metrics = {}
	df = df.copy()

# Do we even need an AVERAGE score of the 2 below values?
	# metrics["avg_mood_fit"] = round(df["mood_score"].mean(), 2)
	# metrics["avg_semantic_match"] = round(df["normalized_similarity"].mean(), 2)

	for feat in features:
		metrics[f"avg_{feat}"] = round(df[feat].mean(), 2)

	metrics["unique_artists"] = df["artist_name"].nunique()
	metrics["unique_genres"] = df["genre"].nunique()
	#metrics["artist_diversity"] = round(df["artist_name"].nunique() / total, 2)
	# metrics["genre_diversity"]  = round(df["genre"].nunique() / total, 2)

	return metrics

if __name__ == "__main__":
	from src.explainer import explain_playlist
	from src.diversity_mmr import deduplicate_by_name, apply_mmr
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
	print(f"Ranked pool size: {len(ranked)} --------> After dedup: {len(dedup_df)}")
	final_result = apply_mmr(dedup_df, intent, lambda_param = 0.7)

	PLAYLIST = explain_playlist(final_result, intent)
	#print(PLAYLIST[["track_name", "artist_name", "final_score", "explanation"]].to_string())
	metrics = evaluate_playlist(final_result, intent)
	for k, v in metrics.items():
		print(f"{k}: {v}")