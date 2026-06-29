import os
import faiss
import time
import tempfile
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from src.config_loader import CFG
from sentence_transformers import SentenceTransformer
from src.logger import get_logger
logger = get_logger(__name__)

from huggingface_hub import hf_hub_download
load_dotenv()

index = None
metadata = None

METADATA_COLUMNS = [
	"track_name", "artist_name", "genre", "popularity",
	"energy", "valence", "acousticness", "instrumentalness",
	"speechiness", "danceability",
]
def _artifact_cache_dir() -> str:
	return os.path.join(tempfile.gettempdir(), "ragnroll")

def load_artifacts():
	global index, metadata
	if index is None or metadata is None:
		cache_dir = _artifact_cache_dir()
		local_index = CFG['indexing']['faiss_index_path']
		local_meta = CFG['data']['metadata_path']

		if os.path.isfile(local_index) and os.path.isfile(local_meta):
			logger.info("Loading artifacts from local paths...")
			index = faiss.read_index(local_index)
			logger.info("FAISS loaded (%d vectors)", index.ntotal)
			metadata = pd.read_parquet(local_meta, columns=METADATA_COLUMNS)
			logger.info("metadata loaded (%d rows)", len(metadata))
		else:
			logger.info("Downloading artifacts from HF Hub...")
			index_path = hf_hub_download(
				repo_id="ReagentGrignard/ragnroll-artifacts",
				filename="faiss.index",
				repo_type="dataset",
				token=os.getenv("HF_KEY"),
				cache_dir=cache_dir,
			)
			meta_path = hf_hub_download(
				repo_id="ReagentGrignard/ragnroll-artifacts",
				filename="track_metadata.parquet",
				repo_type="dataset",
				token=os.getenv("HF_KEY"),
				cache_dir=cache_dir,
			)
			index = faiss.read_index(index_path)
			logger.info("FAISS loaded (%d vectors)", index.ntotal)
			metadata = pd.read_parquet(meta_path, columns=METADATA_COLUMNS)
			logger.info("metadata loaded (%d rows)", len(metadata))
	return index, metadata

start = time.time()
# index = faiss.read_index(CFG['indexing']['faiss_index_path'])
# logger.info("FAISS loaded")
# metadata = pd.read_parquet(CFG['data']['metadata_path'])
# logger.info("metadata loaded")

_model = None

def _get_model() -> SentenceTransformer:
	global _model
	if _model is None:
		logger.info("loading model...")
		_model = SentenceTransformer(
			CFG['indexing']['model_name'],
			device="cpu",
			model_kwargs={"token": os.getenv("HF_KEY")},
		)
		logger.info("model loaded")
	return _model

HARD_FILTERS = ['valence', 'energy']
# SOFT_FILTERS = ['acousticness', 'instrumentalness', 'danceability']
features = ['energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness', 'danceability']

def retrieve(query: str, k:int = 200) -> pd.DataFrame:
	index, metadata = load_artifacts()
	# BGE model requires prefix to signal it to work with the prompt
	prefixed = "Represent this sentence for searching relevant passages: " + query

	encoded_prompt = _get_model().encode(prefixed)		      	# (384, )
	encoded_prompt = encoded_prompt.reshape(1, -1) 				# (1, 384)
	faiss.normalize_L2(encoded_prompt)
	distances, indices = index.search(encoded_prompt, k)

	result_df = metadata.iloc[indices[0]].copy()
	result_df['similarity_score'] = distances[0]
	result_df = result_df[result_df['popularity'] > 15]
	return result_df
 
def filter_songs(songs: pd.DataFrame, intent: dict) -> pd.DataFrame:
	filtered = songs.copy()
	playlist_length = intent.get("playlist_length", 15)

	for feature in HARD_FILTERS:
		if feature not in intent:
			continue
		low, high = intent[feature]
		# expand bounds by 20% to avoid over-filtering. Major problem was occuring here
		margin = (high - low) * 0.2
		before = len(filtered)
		filtered = filtered[filtered[feature].between(max(0.0, low - margin), min(1.0, high + margin))]
		logger.info(f"{feature} [{max(0.0, low - margin):.2f}, {min(1.0, high + margin):.2f}]: {before} -> {len(filtered)}")

	# Fallback: if too few, return everything
	if len(filtered) < playlist_length:
		logger.warning(f"{len(filtered)} after hard filter. Returning full pool.")
		return songs

	logger.info(f"{len(filtered)} songs pass hard filter. {playlist_length} is the playlist length.")
	return filtered

def _label(mid: float) -> str:
    if mid < 0.2:   return "very low"
    if mid < 0.4:   return "low"
    if mid < 0.6:   return "medium"
    if mid < 0.75:  return "high"
    return "very high"

def rebuild_retrieval_query(intent: dict) -> str:
    parts = []
    for f in HARD_FILTERS:
        if f in intent:
            low, high = intent[f]
            if [low, high] == [0.0, 1.0]:  # unconstrained, skip
                continue
            mid = (low + high) / 2
            parts.append(f"{_label(mid)} {f}")
    parts += intent.get("moods", [])
    parts += intent.get("genres", [])										# GENRE ADDED
    #parts += intent.get("activities", [])
    return " ".join(parts)


if __name__ == "__main__":
	from src.parser import parse_intent
	test_prompt =  "Suggest love songs for a techno-funk party"				# a very contradicting prompt
	intent_dict = parse_intent(test_prompt)
	query = rebuild_retrieval_query(intent_dict)
	results = retrieve(query, k=1000)
	filtered = filter_songs(results, intent_dict)

	print(intent_dict)
	print(f"[QUERY] {query}")
	print(filtered[['track_name', 'artist_name', 'genre', 'energy', 'valence', 'similarity_score']].sort_values(by="similarity_score", ascending=False))
	print(f"Songs: {len(results)} --> Filtered: {len(filtered)}")

	print(f"\n\nTime taken: {time.time()-start} \n\n")
