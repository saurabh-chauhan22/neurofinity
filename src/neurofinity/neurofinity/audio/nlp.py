from typer import Typer
from loguru import logger

from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

app = Typer()

class NLPTranscripter:
    '''
    Class to convert the audio transcript to summary, topics, and subtopics.
    '''
    def __init__(self, transcript):
        assert transcript is not None, "Transcript cannot be None"
        self.transcript = transcript
    
    def load_models(self):
        try:
            task_tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")
            task_model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large")
            return task_tokenizer, task_model 
        except Exception as e:
            logger.error(f"Exception while loading models: {e}")
            return None, None
    
    def load_summarizer_model(self):
        return pipeline("summarization", model="facebook/bart-large-cnn")
    
    def chunks_splitting(self):
        try:
            return self.split_text(self.transcript)
        except Exception as e:
            logger.error(f"Text splitting error: {e}")
            return []
    
    @property
    def get_topic_model(self):
        text_size = len(self.transcript.split())    
        return self.get_bertopic_model(text_size) 
    
    def get_bertopic_model(self, text_size):
        assert text_size is not None, "Text size cannot be None"
        if text_size < 50:  # Small dataset (less than 50 words)
            return BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                umap_model=None,  
                hdbscan_model=None,
                vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                min_topic_size=1,  
                nr_topics="auto"
            )
        else:  
            return BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                min_topic_size=2,  # Ensures valid clusters
                nr_topics="auto"
            )
        
    @staticmethod
    def split_text(text, max_words=500):
        words = text.split()
        if not words:
            raise ValueError("Empty transcript")
        return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    
    def summarize_text(self, chunks):
        if not chunks:
            return "No text available"
        summaries = []
        summarizer = self.load_summarizer_model()
        
        for chunk in chunks:
            try:
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Summarization error: {e}")
        return " ".join(summaries)
    
    def extract_tasks(self, text):
        task_tokenizer, task_model = self.load_models()
        if not task_tokenizer or not task_model:
            return "Failed to load models for task extraction."
        
        if not text.strip():
            return "No summary available for task extraction."
        try:
            question = "What are the action items from the meeting?"
            input_text = f"{question} {text}"
            input_ids = task_tokenizer.encode(input_text, return_tensors="pt")
            output = task_model.generate(input_ids)
            return task_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Task extraction error: {e}")
            return "Error extracting tasks."

    def extract_topics(self, summary):
        if not summary or len(summary.split()) < 10:
            return "Not enough data."

        sentences = [s.strip() for s in summary.split(". ") if len(s.strip()) > 5]
        
        if len(sentences) < 3:
            return "Not enough sentences."

        try:
            topics, _ = self.get_topic_model.fit_transform(sentences) 
            return self.get_topic_model.get_topic_info()
        except ValueError as e:
            logger.error(f"Topic error: {e}")
            return "Topic extraction failed."
        except Exception as e:
            logger.error(f"Exception error: {e}")
            return "Unexpected error in topic extraction."
            
    def task_extraction(self):
        '''
        Get the extracted topics, summary, and subtopics.
        '''
        chunks = self.chunks_splitting()
        summary = self.summarize_text(chunks)
        tasks = self.extract_tasks(self.transcript)
        topics_info = self.extract_topics(summary)
        
        return {"summary": summary, "tasks": tasks, "topics": topics_info}