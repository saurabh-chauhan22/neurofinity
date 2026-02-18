import requests
import os
import time

from typer import Typer
from loguru import logger

from neurofinity.config import ENGLISH_MODEL_API_URL
 
app = Typer()
class AudioToText:
    '''
    Class to extract audio to text from AI model from hugging face API
    '''
    def __init__(self,audio_path,model_id="qwen2-audio-7b-instruct"):
        '''
        constructor for audio path
        '''
        assert model_id is not None
        
        self.model_id = model_id    
        self.audio_path = audio_path

    @property
    def headers(self)->dict:
        return {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}",
    "Content-Type": "audio/wav"  }
       
    def transcript_api_call(self):
        '''
        get the transcript api call from hugging face
        '''
        with open(os.path.abspath(self.audio_path), "rb") as f:
            audio_data = f.read()
        transcript = requests.post(ENGLISH_MODEL_API_URL, headers=self.headers, data=audio_data)
        return transcript


    def output_text(self):
        transcript = self.transcript_api_call()
        if transcript.status_code == 200: 
            time.sleep(5)
            logger.info("Transcript:", transcript.json())
        else:
            logger.error(f"Request failed with status code {transcript.status_code}: {transcript.text}")
        return transcript   

# emotion_transcript = requests.post(EMOTION_MODEL_API_URL, headers=headers, data=audio_data)

