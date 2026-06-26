import pandas as pd
from config_loader import CFG
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
	template="""
	You are explaining why a song was selected for a playlist.

	User intent: moods={moods}, activities={activities}
	Song: "{track_name}" by {artist_name} ({genre})
	Audio features: energy={energy}, valence={valence}, acousticness={acousticness}, instrumentalness={instrumentalness}, speechiness={speechiness}
	Mood fit: {mood_score} | Semantic match: {normalized_similarity} | Final_score : {final_score}

	Write exactly ONE sentence (max 20 words) explaining why this song fits the intent.
	Use the actual feature values. No fluff. No preamble. Just the sentence
	""",
	input_variables= [
        "track_name", "artist_name", "genre", "moods", "activities",
        "energy", "valence", "acousticness", "instrumentalness", "speechiness",
        "mood_score", "normalized_similarity", "final_score"
    	]
)

def _explain_song(row: pd.Series, intent: dict) -> str:
	llm = OllamaLLM(model=CFG['retrieval']['llm_model'], temperature=CFG['retrieval']['llm_temperature'])
	chain = prompt | llm | StrOutputParser()

	try:
		return chain.invoke({
			"track_name":        row["track_name"],
            "artist_name":       row["artist_name"],
            "genre":             row.get("genre", "unknown"),
            "moods":             ", ".join(intent.get("moods", [])),
            "activities":        ", ".join(intent.get("activities", [])),
            "energy":            round(float(row["energy"]), 2),
            "valence":           round(float(row["valence"]), 2),
            "acousticness":      round(float(row["acousticness"]), 2),
            "instrumentalness":  round(float(row["instrumentalness"]), 2),
            "speechiness":       round(float(row["speechiness"]), 2),
            "mood_score":    	 round(float(row["mood_score"]), 2),
            "normalized_similarity":   round(float(row["normalized_similarity"]), 2),
			"final_score":   	 round(float(row["final_score"]), 2)
        })
	except Exception as e:
		return f"Explanation unavailable: {e}"

def explain_playlist(final_result: pd.DataFrame, intent: dict) -> pd.DataFrame:
	df = final_result.copy()
	df["explanation"] = df.apply(lambda row: _explain_song(row, intent), axis=1)
	return df


if __name__ == "__main__":
	from src.diversity_mmr import deduplicate_by_name, apply_mmr
	from src.ranker import rank_songs
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent

	print("Imports COMPLETED")
	test_prompt =  "A classy, slow Italian playlist with mafia-vibes for deep thinking"
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
	print(PLAYLIST[["track_name", "artist_name", "final_score", "explanation"]].to_string())
