import os
import pandas as pd
from dotenv import load_dotenv
from src.config_loader import CFG
#from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
	template="""
	You are explaining why a playlist was generated for a user.

	User intent: moods={moods}, activities={activities}, avoid={avoid}
	Playlist contains {playlist_length} songs across these genres: {genres}
	Average audio features: energy={energy}, valence={valence}, acousticness={acousticness}

	Write 2-3 sentences explaining how this playlist fits the user's intent.
	Be specific, with convincing reasoning. Reference the moods, energy level, and genre mix. No fluff.
	""",
	input_variables=[
		"moods", "activities", "avoid", "playlist_length",
		"genres", "energy", "valence", "acousticness",
	],
)

llm = ChatGroq(
	model=CFG['retrieval']['llm_model'],
	temperature=CFG['retrieval']['llm_temperature'],
	api_key=os.getenv("GROQ_API_KEY"),
)
chain = prompt | llm | StrOutputParser()

def explain_playlist(final_result: pd.DataFrame, intent: dict) -> str:
	try:
		return chain.invoke({
			"moods": ", ".join(intent.get("moods", [])),
			"activities": ", ".join(intent.get("activities", [])),
			"avoid": ", ".join(intent.get("avoid", [])),
			"playlist_length": len(final_result),
			"genres": ", ".join(final_result["genre"].value_counts().head(5).index.tolist()),
			"energy": round(final_result["energy"].mean(), 2),
			"valence": round(final_result["valence"].mean(), 2),
			"acousticness": round(final_result["acousticness"].mean(), 2),
		})
	except Exception as e:
		return f"Explanation unavailable: {e}"


if __name__ == "__main__":
	from src.diversity_mmr import deduplicate_by_name, apply_mmr
	from src.ranker import rank_songs
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent

	print("Imports COMPLETED")
	test_prompt = "A classy, slow Italian playlist with mafia-vibes for deep thinking"
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

	final_result = apply_mmr(dedup_df, intent, lambda_param=0.7)

	explanation = explain_playlist(final_result, intent)
	print(explanation)
	print(final_result[["track_name", "artist_name", "genre"]].to_string())
