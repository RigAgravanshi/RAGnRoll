import pandas as pd
import re 

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

if __name__ == "__main__":
	from src.ranker import rank_songs
	print("RANKER IMPORTED")
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent
	print("Imports COMPLETED")
	test_prompt = "Suggest love songs fitting a techno-funk party"
	intent = parse_intent(test_prompt)
	print("PARSER.PY DONE")
	query = rebuild_retrieval_query(intent)
	print("QUERY REBUILT DONE")
	results = retrieve(query, k=1000)
	print("RETRIEVAL DONE")
	filtered = filter_songs(results, intent)
	print("\nFILTERING ALSO DONE!! Now Ranking......\n")
	ranked = rank_songs(filtered, intent)
	print("Ranking of Kings DONE")
	dedup_df = deduplicate_by_name(ranked)

	print(f"Ranked pool size: {len(ranked)}")
	print(f"After dedup: {len(dedup_df)}")
	print(dedup_df[['track_name', 'norm_name', 'final_score']].head(20))
