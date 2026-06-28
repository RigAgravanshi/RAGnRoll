import pandas as pd
from src.config_loader import CFG
from src.retriever import features

def mood_fit_score(row, intent: dict):
	feature_score = []
	for feature in features:
		if feature not in intent:
			continue
		low, high = intent[feature]
		mid = (low + high) / 2
		distance = abs(row[feature] - mid)
		feature_score.append(1 - distance)

	if not feature_score:
		return 0.0
	score = sum(feature_score)/len(feature_score)
	return score

def score_normalization(similarity_score: pd.Series) -> pd.Series:
	min_val = similarity_score.min()
	max_val = similarity_score.max()
	if min_val == max_val:
		return pd.Series([1.0]* len(similarity_score), index = similarity_score.index)
	return (similarity_score - min_val) / (max_val - min_val)

def rank_songs(filtered_songs: pd.DataFrame, intent: dict) -> pd.DataFrame:
	ranked = filtered_songs.copy()
	ranked["mood_score"] = ranked.apply(lambda row: mood_fit_score(row, intent), axis=1)
	ranked["normalized_similarity"] = score_normalization(ranked["similarity_score"])
	ranked["final_score"] = CFG["ranking"]["mood_score_weight"]*ranked["mood_score"] + CFG["ranking"]["similarity_weight"]*ranked["normalized_similarity"]

	ranked = ranked.sort_values(by="final_score", ascending=False)
	return ranked														
if __name__ == "__main__":
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent
	print("Imports COMPLETED")
	test_prompt = "Italiano playlist for when your plan has succeeded against your enemies"
	intent = parse_intent(test_prompt)
	print("PARSER.PY DONE")
	query = rebuild_retrieval_query(intent)
	print("QUERY REBUILT DONE")
	results = retrieve(query, k=1000)
	print("RETRIEVAL DONE")
	filtered = filter_songs(results, intent)
	print("\nFILTERING ALSO DONE!!\n")

	ranked = rank_songs(filtered, intent)
	print(ranked[['track_name', 'artist_name', 'final_score', 'popularity']])
		