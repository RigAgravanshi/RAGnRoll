import os
import gc
import faiss
import time
import tempfile
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from src.config_loader import CFG
from src.logger import get_logger
logger = get_logger(__name__)

from huggingface_hub import hf_hub_download
load_dotenv()

index = None
metadata = None

features = ["energy", "valence", "acousticness", "instrumentalness", "speechiness", "danceability"]
METADATA_COLUMNS = ["track_name", "artist_name", "genre", "popularity", *features]
HF_REPO = "ReagentGrignard/ragnroll-artifacts"
_FAISS_MMAP_FLAGS = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY


def _artifact_cache_dir() -> str:
	return os.path.join(tempfile.gettempdir(), "ragnroll")


def _read_faiss_index(path: str):
	try:
		idx = faiss.read_index(path, _FAISS_MMAP_FLAGS)
		logger.info("FAISS loaded with mmap (%d vectors)", idx.ntotal)
		return idx
	except Exception as exc:
		logger.warning("FAISS mmap unavailable (%s); loading into RAM", exc)
		idx = faiss.read_index(path)
		logger.info("FAISS loaded (%d vectors)", idx.ntotal)
		return idx


def _compact_metadata(df: pd.DataFrame) -> pd.DataFrame:
	df = df[METADATA_COLUMNS].copy()
	df["genre"] = df["genre"].astype("category")
	for col in features:
		df[col] = df[col].astype("float32")
	df["popularity"] = pd.to_numeric(df["popularity"], downcast="integer")
	return df


def _read_metadata(path: str) -> pd.DataFrame:
	logger.info("Loading metadata from %s", os.path.basename(path))
	df = pd.read_parquet(path, columns=METADATA_COLUMNS, engine="pyarrow")
	df = _compact_metadata(df)
	logger.info("metadata loaded (%d rows)", len(df))
	return df


def _resolve_metadata_path(cache_dir: str) -> str:
	local_slim = CFG["data"].get("metadata_slim_path", "")
	local_full = CFG["data"]["metadata_path"]
	if local_slim and os.path.isfile(local_slim):
		return local_slim
	if os.path.isfile(local_full):
		return local_full

	for filename in ("track_metadata_slim.parquet", "track_metadata.parquet"):
		try:
			return hf_hub_download(
				repo_id=HF_REPO,
				filename=filename,
				repo_type="dataset",
				token=os.getenv("HF_KEY"),
				cache_dir=cache_dir,
			)
		except Exception as exc:
			logger.warning("HF metadata file %s unavailable: %s", filename, exc)
	raise FileNotFoundError("No metadata parquet found locally or on Hugging Face Hub")


def _resolve_index_path(cache_dir: str) -> str:
	local_index = CFG["indexing"]["faiss_index_path"]
	if os.path.isfile(local_index):
		return local_index
	return hf_hub_download(
		repo_id=HF_REPO,
		filename="faiss.index",
		repo_type="dataset",
		token=os.getenv("HF_KEY"),
		cache_dir=cache_dir,
	)


def load_artifacts():
	global index, metadata
	if index is not None and metadata is not None:
		return index, metadata

	cache_dir = _artifact_cache_dir()
	index_path = _resolve_index_path(cache_dir)
	meta_path = _resolve_metadata_path(cache_dir)

	index = _read_faiss_index(index_path)
	metadata = _read_metadata(meta_path)
	gc.collect()
	return index, metadata

start = time.time()

_model = None


def _get_model():
	global _model
	if _model is None:
		from sentence_transformers import SentenceTransformer
		logger.info("loading model...")
		_model = SentenceTransformer(
			CFG["indexing"]["model_name"],
			device="cpu",
			model_kwargs={"token": os.getenv("HF_KEY")},
		)
		logger.info("model loaded")
	return _model

HARD_FILTERS = ["valence", "energy"]

def retrieve(query: str, k: int = 200) -> pd.DataFrame:
	index, metadata = load_artifacts()
	# BGE model requires prefix to signal it to work with the prompt
	prefixed = "Represent this sentence for searching relevant passages: " + query

	encoded_prompt = _get_model().encode(prefixed)		      	# (384, )
	encoded_prompt = encoded_prompt.reshape(1, -1) 				# (1, 384)
	faiss.normalize_L2(encoded_prompt)
	distances, indices = index.search(encoded_prompt, k)

	result_df = metadata.iloc[indices[0]].copy()
	result_df["similarity_score"] = distances[0]
	return result_df.loc[result_df["popularity"] > 15].reset_index(drop=True)


def filter_songs(songs: pd.DataFrame, intent: dict) -> pd.DataFrame:
	filtered = songs
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
		logger.warning("%d after hard filter. Returning full pool.", len(filtered))
		return songs

	logger.info("%d songs pass hard filter. playlist_length=%d", len(filtered), playlist_length)
	return filtered


def _label(mid: float) -> str:
	if mid < 0.2:
		return "very low"
	if mid < 0.4:
		return "low"
	if mid < 0.6:
		return "medium"
	if mid < 0.75:
		return "high"
	return "very high"


def rebuild_retrieval_query(intent: dict) -> str:
	parts = []
	for f in HARD_FILTERS:
		if f in intent:
			low, high = intent[f]
			if [low, high] == [0.0, 1.0]:
				continue
			mid = (low + high) / 2
			parts.append(f"{_label(mid)} {f}")
	parts += intent.get("moods", [])
	parts += intent.get("genres", [])
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
	print(
		filtered[["track_name", "artist_name", "genre", "energy", "valence", "similarity_score"]]
		.sort_values(by="similarity_score", ascending=False)
	)
	print(f"Songs: {len(results)} --> Filtered: {len(filtered)}")
	print(f"\n\nTime taken: {time.time() - start} \n\n")
