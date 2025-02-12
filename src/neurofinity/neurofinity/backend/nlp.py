from typer import Typer
from loguru import logger

import textwrap
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

app = Typer()
class NLPTransacripter:
    '''
    Class to convert the audio transacript to summarry, topics and subtopics.
    '''
    def __init__(self, transcript):
        assert transcript is not None
        self.transcript = transcript
    
    def load_models(self):
        try:
            task_tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")
            task_model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large")
        except Exception as e:
            print(f"exception: {e}")
        return task_tokenizer, task_model 
    
    def load_summizer_model(self):
        return pipeline("summarization", model="facebook/bart-large-cnn")
    
    def chunks_splitting(self):
        chunks = list()
        try:
            chunks = self.split_text(self.transcript)
        except Exception as e:
            logger.error(f"Text splitting error: {e}")
        return chunks

    
    @property
    def get_topic_model(self):
        text_size = len(self.transcript.split())    
        return self.get_bertopic_model(text_size) 
    
        
    def get_bertopic_model(self, text_size):
        assert text_size is not None
        if text_size < 50:  # Small dataset (less than 50 words)
            return BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                umap_model=None,  # Disable dimensionality reduction
                hdbscan_model=None,  # Avoids clustering errors
                vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                min_topic_size=1,  # Allows small topic groups
                nr_topics="auto"
            )
        else:  # Large dataset (more than 50 words)
            return BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                min_topic_size=2,  # Ensures valid clusters
                nr_topics="auto"
            )
        
    def split_text(text, max_words=500):
        words = text.split()
        if len(words) == 0:
            raise ValueError("Empty transcript")
        return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

        
    def summarize_text(self,chunks):
        if not chunks:
            return "No text available"
        summaries = list()
        summerizer= self.load_summizer_model()
        
        for chunk in chunks:
            try:
                summary = summerizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Summarization error: {e}")
        return " ".join(summaries)

    def extract_tasks(self,text):
        task_tokenizer, task_model = self.load_models()
        if not text or text.strip() == "":
            return "No summary available for task extraction."
        try:
            question = "What are the action items from the meeting?"
            input_text = f"{question} {text}"
            input_ids = task_tokenizer.encode(input_text, return_tensors="pt")
            output = task_model.generate(input_ids)
            return task_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Task extraction error: {e}")


    def extract_topics(self,summery):
        if not summery or len(summery.split()) < 10:  # Ensure enough words
            return "Not enough data."

        sentences = [s.strip() for s in summery.split(". ") if len(s.strip()) > 5]

        if len(sentences) < 3:  # Ensure enough sentences
            return "Not enough sentences."

        try:
            topics, _ = self.get_topic_model.fit_transform(sentences) 
            return self.get_topic_model.get_topic_info()
        except ValueError as e:
            logger.error(f"Topic error ",e)
        except Exception as e:
            logger.error(f"Exception error ",e)
            
            
    def task_extraction(self):
        '''
        Get the extracted topics summary and sub topics 
        '''
        chunks = self.chunks_splitting()
        summary = self.summarize_text(chunks)
        tasks = self.extract_tasks()
        topics = self.extract_topics(summary)
        output_dict = {"summary":summary, tasks:topics.Representative_Docs}
        return output_dict



