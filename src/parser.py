import time

import httpx
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from ollama import ResponseError
from src.config_loader import CFG

_OLLAMA_RETRY_ERRORS = (ResponseError, httpx.ConnectError, ConnectionError)

def parse_intent(user_prompt: str, max_retries: int = 3) -> dict:
	llm = OllamaLLM(model=CFG['retrieval']['llm_model'])
	chain = prompt_template | llm | JsonOutputParser()
	for attempt in range(1, max_retries + 1):
		try:
			return chain.invoke({'user_prompt': user_prompt})
		except _OLLAMA_RETRY_ERRORS as e:
			if attempt == max_retries:
				raise
			wait = 2 * attempt
			print(f"[PARSER] Ollama error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
			time.sleep(wait)

prompt_template = PromptTemplate(
	template = """
	You are a music mood analysis engine.
	Given a user playlist prompt, return ONLY a JSON object. No explanation. No markdown.
	
	JSON schema:
	{{
	"moods": [list of mood strings],
	"activities": [list of activity strings],
	"energy": [min, max],
	"valence": [min, max],
	"acousticness": [min, max],
	"instrumentalness": [min, max],
	"speechiness": [min, max],
	"danceability": [min, max],
	"avoid": [list of genres/styles to avoid],
	"playlist_length": integer
	}}

	Rules:
	- All audio feature values are floats between 0.0 and 1.0
	- valence: 0.0 = sad/dark/melancholic, 1.0 = happy/euphoric/upbeat. Calm/focused playlists are NOT high valence.
	- energy: 0.0 = silent/still, 1.0 = intense/loud. Focused work playlists are low-medium energy.
	- instrumentalness: 0.0 = has lyrics, 1.0 = purely instrumental. Classical/ambient = high instrumentalness.
	- If a feature is unconstrained, return [0.0, 1.0]
	- If no playlist length specified, default to 15
	- avoid list should contain lowercase genre/style strings. 
	- Use double quotes for all strings, including inside lists. Single quotes are invalid JSON.
	- Return raw JSON only, nothing else

	Example:
	Prompt: "classical music for deep focus, deadline work, no lyrics"
	Output:
	{{
	"moods": ["focused", "calm", "concentrated"],
	"activities": ["studying", "working"],
	"energy": [0.1, 0.5],
	"valence": [0.2, 0.55],
	"acousticness": [0.7, 1.0],
	"instrumentalness": [0.8, 1.0],
	"speechiness": [0.0, 0.1],
	"danceability": [0.0, 0.3],
	"avoid": ["electronic", "rock", "pop"],
	"playlist_length": 15
	}}

	Prompt: "upbeat happy summer road trip playlist"
	Output:
	{{
	"moods": ["happy", "energetic", "euphoric"],
	"activities": ["road trip", "driving", "partying"],
	"energy": [0.7, 1.0],
	"valence": [0.7, 1.0],
	"acousticness": [0.0, 0.4],
	"instrumentalness": [0.0, 0.3],
	"speechiness": [0.0, 0.6],
	"danceability": [0.6, 1.0],
	"avoid": ["classical", "ambient", "sad"],
	"playlist_length": 15
	}}

	Prompt: "calm late night coding playlist of 20 songs"
	Output: 
	{{
	"moods": ["calm", "focused"],
	"activities": ["coding", "studying", "research"],
	"energy": [0.1, 0.45],
	"valence": [0.2, 0.55],
	"acousticness": [0.3, 1.0],
  	"instrumentalness": [0.2, 1.0],
	"speechiness": [0.0, 0.30],
	"danceability": [0.0, 0.35],
	"avoid": ["hip-hop", "rap"],
	"playlist_length": 20
	}}

	Prompt: "high energy gym workout"
	Output: 
	{{
	"activities": ["workout", "focus", "sports"],
	"energy": [0.7, 1.0],
	"valence": [0.5, 1.0],
	"speechiness": [0.0, 0.7],
	"acousticness": [0.0, 0.3],
	"danceability": [0.6, 1.0],
	"avoid": ["classical", "ambient"],
	"playlist_length": 15
	}}

	Prompt: "dark cinematic dramatic tense playlist for deep thinking"
	Output:
	{{
	"moods": ["dramatic", "thrilling", "dark", "cinematic"],
	"activities": ["deep thinking", "writing"],
	"energy": [0.2, 0.5],
	"valence": [0.1, 0.4],
	"acousticness": [0.3, 0.9],
	"instrumentalness": [0.6, 1.0],
	"speechiness": [0.0, 0.2],
	"danceability": [0.0, 0.3],
	"avoid": ["pop", "hip-hop"],
	"playlist_length": 15
	}}

	User Prompt: {user_prompt}
	""",
    input_variables=['user_prompt']
)


if __name__ == "__main__":
	example1 = "A classy, slow Italian playlist with mafia-vibes for deep thinking. Around 9-10 songs"
	example2 = "A classical music playlist to lock-in when u r on a deadline"
	example3 = "Suggest songs that would fit a cyberpunk movie set in India"
	example4 = "I want songs for a late-night train journey after a difficult breakup."
	user_prompt = example1
	result = parse_intent(user_prompt)
	print(result)