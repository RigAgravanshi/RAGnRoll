import streamlit as st
st.set_page_config(
	page_title="RAG-n-Roll",
	layout="wide"
	)
st.header("RAG-n-Roll 🪩🎸🎧", divider=True, text_alignment="center")

try:																			# Pls Don't disturb import order. Otherwise Expect Crashes
	from src.evaluation import evaluate_playlist
	from src.explainer import explain_playlist
	from src.diversity_mmr import deduplicate_by_name, apply_mmr
	from src.ranker import rank_songs
	from src.retriever import retrieve, filter_songs, rebuild_retrieval_query
	from src.parser import parse_intent
except ImportError as e:
	st.error(f"Failed to Import Modules {e}", icon="🚨")
	st.stop()


user_prompt = st.text_area(
		label="Enter what You Feel~~", 
		placeholder="e.g. Suggest songs that would fit a cyberpunk movie set in India", 
		help = """
		Describe a Mood/ an Activity/ a Time but don't be vague. 
		Ranking & Scoring is done on various audio features like:
		Energy, Valence, Acousticness, Instrumentalness...etc.

		These are extracted from the Natural Language of your prompt by an LLM
		"""
		)
playlist_length = st.slider(label = "Playlist Length", min_value=5, max_value = 20, step = 1, value = 10)

# Session initialization and Playlist generation: 
for key in ["playlist", "metrics"]:
	if key not in st.session_state:
		st.session_state[key] = None

if st.button("Generate Playlist 🎵"):
	if user_prompt.strip() == '':												# For Empty Prompt/ Blank Space
		st.warning("Enter a prompt first.")
	else:
		try:
			with st.spinner("Parsing your vibe..."):
				intent = parse_intent(user_prompt)
				intent["original_prompt"] = user_prompt
				intent["playlist_length"] = playlist_length
				query = rebuild_retrieval_query(intent)

			with st.spinner("Retrieving Songs from Database"):
				results = retrieve(query)
				filtered = filter_songs(results, intent)

			with st.spinner("Ranking Songs...."):
				ranked = rank_songs(filtered, intent)
				dedup_df = deduplicate_by_name(ranked)
				final_result = apply_mmr(dedup_df, intent)

			with st.spinner("Generating explanations..."):
				PLAYLIST = explain_playlist(final_result, intent)
				metrics = evaluate_playlist(PLAYLIST, intent)

			st.session_state["playlist"] = PLAYLIST
			st.session_state["metrics"] = metrics
		
		except Exception as e:
			st.error(f"Pipeline Error: {e}")
			st.stop()
		
# Playlist has been generated, now to display it......
if st.session_state["playlist"] is not None:
	df = st.session_state["playlist"]
	metrics = st.session_state["metrics"]
	display_df = df.copy()

	st.subheader("Playlist Songs")
	display_cols = ["track_name", "artist_name", "genre", "final_score", "explanation"]
	st.dataframe(
		display_df[display_cols], 
		hide_index=True, width='stretch',
		height=500,
		column_config={
			"final_score" : st.column_config.NumberColumn("Score", format="%.2f", 
				help = "Final score, calculated as Weighted Avg. of Mood-Fit-score and Semantic-Similarity(cosine)"),
			"explanation" : st.column_config.TextColumn("Why", width = "large",
				help = "Generated as per audio features, moods and activities that best match your prompt")
		}
	)

	st.divider()
	st.subheader("Playlist Metrics") 
	col1, col2, col3, col4, col5 = st.columns(5)
	col1.metric("Avg Energy", metrics.get("avg_energy"))
	col2.metric("Avg Valence", metrics.get("avg_valence"))
	col3.metric("Avg Instrumentalness", metrics.get("avg_instrumentalness"))
	col4.metric("Unique Artists", metrics.get("unique_artists"))
	col5.metric("Unique Genres", metrics.get("unique_genres"))
