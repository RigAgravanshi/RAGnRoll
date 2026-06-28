import time

import httpx
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from ollama import ResponseError
from src.config_loader import CFG
from src.logger import get_logger
logger = get_logger(__name__)

_OLLAMA_RETRY_ERRORS = (ResponseError, httpx.ConnectError, ConnectionError)

def parse_intent(user_prompt: str, max_retries: int = 3) -> dict:
	llm = OllamaLLM(model=CFG['retrieval']['llm_model'], temperature=CFG['retrieval']['llm_temperature'])
	chain = prompt_template | llm | JsonOutputParser()
	for attempt in range(1, max_retries + 1):
		try:
			return chain.invoke({'user_prompt': user_prompt})
		except _OLLAMA_RETRY_ERRORS as e:
			if attempt == max_retries:
				raise
			wait = 2 * attempt
			logger.error(f"Ollama error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
			time.sleep(wait)

prompt_template = PromptTemplate(
	template = """
	You are a music mood analysis engine.
	Given a user playlist prompt, return ONLY a JSON object. No explanation. No markdown.
	
	JSON schema:
	{{
	"moods": [list of mood strings],
	"genres": [list of target genre strings],
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
	- avoid list should contain lowercase genre strings. 
	- Use double quotes for all strings, including inside lists. Single quotes are invalid JSON.
	- Return raw JSON only, nothing else
	- genres: ONLY extract if explicitly stated or strongly implied by cultural reference.
  	- genre values MUST be from this exact list:
		acoustic, afrobeat, alt-rock, ambient, black-metal, blues, breakbeat, cantopop, 
		chicago-house, chill, classical, club, comedy, country, dance, dancehall, death-metal, 
		deep-house, detroit-techno, disco, drum-and-bass, dub, dubstep, edm, electro, electronic, 
		emo, folk, forro, french, funk, garage, german, gospel, goth, grindcore, groove, guitar, 
		hard-rock, hardcore, hardstyle, heavy-metal, hip-hop, house, indian, indie-pop, industrial, 
		jazz, k-pop, metal, metalcore, minimal-techno, new-age, opera, party, piano, pop, pop-film, 
		power-pop, progressive-house, psych-rock, punk, punk-rock, rock, rock-n-roll, romance, sad, 
		salsa, samba, sertanejo, show-tunes, singer-songwriter, ska, sleep, songwriter, soul, 
		spanish, swedish, tango, techno, trance, trip-hop
	Do NOT infer genre from mood alone. Return [] if uncertain.

	Example:
	Prompt: "classical music for deep focus, deadline work, no lyrics"
	Output:
	{{
	"moods": ["focused", "calm", "concentrated"],
	"genres": ["classical", "ambient", "new-age", "piano"],
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

	Prompt: "upbeat happy summer road trip playlist."
	Output:
	{{
	"moods": ["happy", "energetic", "euphoric"],
	"genres": ["pop", "indie-pop", "rock", "pop-film"],
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
	"genres": [],
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
	"genres": ["electronic", "hip-hop", "hard-rock", "edm", "dance"],
	"energy": [0.7, 1.0],
	"valence": [0.5, 1.0],
	"speechiness": [0.0, 0.7],
	"acousticness": [0.0, 0.3],
	"danceability": [0.6, 1.0],
	"avoid": ["classical", "ambient"],
	"playlist_length": 15
	}}

	Prompt: "romantic valentine's day playlist for a candlelit dinner"
	Output:
	{{
	"moods": ["romantic", "intimate", "warm"],
	"genres": ["romance", "soul", "jazz", "acoustic", "singer-songwriter"],
	"activities": ["dining", "date night"],
	"energy": [0.2, 0.55],
	"valence": [0.5, 0.8],
	"acousticness": [0.3, 0.9],
	"instrumentalness": [0.0, 0.5],
	"speechiness": [0.0, 0.3],
	"danceability": [0.2, 0.6],
	"avoid": ["edm", "metal", "hip-hop", "electronic"],
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
	example5 = "ominous music like where u r a spy who is on an important missions"
	user_prompt = example5
	result = parse_intent(user_prompt)
	print(result)