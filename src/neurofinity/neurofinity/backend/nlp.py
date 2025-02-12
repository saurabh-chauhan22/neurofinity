import textwrap
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


meeting_transcript = """
Topic: Implementing a Customer Data Analytics Pipeline
Moderator (Alice):
Good morning, everyone. Today’s meeting is focused on implementing a new customer data analytics pipeline. Our goal is to process and analyze customer data efficiently to improve marketing campaigns. We will discuss requirements, architecture, development steps, and deadlines. Let’s start with an overview.

Project Manager (Bob):
Thanks, Alice. Our client has requested an automated data pipeline to process customer transactions, website interactions, and demographic data. The data will be used for customer segmentation, predictive modeling, and targeted marketing.

Currently, the marketing team manually extracts data from different sources, which is time-consuming. The new pipeline will automate this process, integrate various data sources, and generate actionable insights.

Step 1: Data Collection and Integration
Data Engineer (Charlie):
The first step is data collection. We need to ingest data from multiple sources, including:

Transactional databases (customer purchases, refunds, etc.)
Web analytics logs (website visits, page interactions)
CRM systems (customer profiles, complaints, and feedback)
External data providers (demographics, purchasing behavior)
The pipeline will use Apache Kafka for real-time streaming and AWS S3 for batch processing.

Action Items:
Identify relevant data sources (Deadline: Next Monday)
Set up Kafka topics for real-time ingestion (Deadline: Next Friday)

Step 2: Data Processing & Cleaning
Senior Data Scientist (David):
Once we have the data, we need to clean and normalize it. Some common issues include missing values, duplicate records, and inconsistent formats.

We will use PySpark for distributed processing and implement data validation rules to handle:

Null values → Impute missing data using statistical methods.
Duplicate entries → Remove based on unique customer ID.
Incorrect formats → Standardize date formats and phone numbers.
Action Items:
Define data validation rules (Deadline: 2 weeks)
Implement initial ETL pipeline (Deadline: 3 weeks)

Step 3: Data Storage & Management
Database Architect (Eve):
For storage, we will use Amazon Redshift as the primary data warehouse. It will store structured data, while AWS S3 will store raw event logs.

We also need data partitioning to optimize query performance. The schema will be star-schema-based, including:

Fact tables: Customer transactions, website visits
Dimension tables: Customer demographics, campaign interactions
Action Items:
Finalize database schema (Deadline: Next Wednesday)
Set up Redshift clusters (Deadline: 4 weeks)

Step 4: Data Analysis & Model Training
Lead Data Analyst (Frank):
Once the pipeline is set up, we will focus on data analysis and predictive modeling. Our goals include:

Customer segmentation → Identify high-value customers
Churn prediction → Forecast customer attrition risk
Recommendation engine → Suggest products based on behavior
We will use Python, Pandas, and Scikit-learn for initial data exploration and model development.

Action Items:
Define key KPIs (Deadline: Next Friday)
Train first predictive model (Deadline: 5 weeks)

Step 5: Dashboard & Reporting
BI Developer (Grace):
The final step is to build a dashboard for real-time monitoring. The dashboard will include:

Customer engagement reports
Conversion funnel analysis
Campaign effectiveness metrics
We will use Tableau and Power BI for visualization and integrate with Redshift.

Action Items:
Design dashboard layout (Deadline: Next Monday)
Implement data connections (Deadline: 6 weeks)

Closing Discussion & Next Steps
Moderator (Alice):
Thanks, everyone. To summarize:

Charlie will handle data ingestion
David will clean and process data
Eve will set up storage
Frank will build predictive models
Grace will create dashboards
Let’s meet next Wednesday for a progress update. Please ensure your teams adhere to the deadlines. Thanks, everyone!

"""

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    task_tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")
    task_model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large")
except Exception as e:
    print(f"exception: {e}")

def get_bertopic_model(text_size):
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

## split large text
def split_text(text, max_words=500):
    words = text.split()
    if len(words) == 0:
        raise ValueError("Empty transcript")
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

try:
    chunks = split_text(meeting_transcript)
except Exception as e:
    print(f"Text splitting error: {e}")
    chunks = []

# step 1: summarize text
def summarize_text(chunks):
    if not chunks:
        return "No text available"
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Summarization error: {e}")
    return " ".join(summaries)

# step 2: extract action items
def extract_tasks(text):
    if not text or text.strip() == "":
        return "No summary available for task extraction."
    try:
        question = "What are the action items from the meeting?"
        input_text = f"{question} {text}"
        input_ids = task_tokenizer.encode(input_text, return_tensors="pt")
        output = task_model.generate(input_ids)
        return task_tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Task extraction error: {e}"


# step 3: extract topics
def extract_topics(text):
    if not text or len(text.split()) < 10:  # Ensure enough words
        return "Not enough data."

    # Convert transcript into a list of meaningful sentences
    sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 5]

    if len(sentences) < 3:  # Ensure enough sentences
        return "Not enough sentences."

    try:
        topics, _ = topic_model.fit_transform(sentences)  # Ensure proper list input
        return topic_model.get_topic_info()
    except ValueError as e:
        return f"Topic modeling error: {str(e)}"
    except Exception as e:
        return f"Unexpected error in topic modeling: {str(e)}"


text_size = len(meeting_transcript.split())  # Count words in transcript
topic_model = get_bertopic_model(text_size)  # Load the best model dynamically

## print summary
summary = summarize_text(chunks)
print("**SUMMARY**", summary)

## print action items
tasks = extract_tasks(summary)
print("**ACTION ITEMS**", tasks)

## print tasks
topics = extract_topics(summary)
print("**TOPICS**", topics.Representative_Docs)


output_dict = {"summary":summary, tasks:topics.Representative_Docs}
print(output_dict)

