
# import pandas as pd
# from googleapiclient.discovery import build
# from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Set up YouTube API
# api_key = ''
# youtube = build('youtube', 'v3', developerKey=api_key)

# # Function to get video details
# def get_video_details(query, max_results):
#     request = youtube.search().list(
#         q=query,
#         part='snippet',
#         type='video',
#         maxResults=max_results
#     )
#     response = request.execute()
#     return response

# # Function to get video transcript
# def get_video_transcript(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         return ' '.join([t['text'] for t in transcript])
#     except NoTranscriptFound:
#         logging.warning(f"Could not fetch transcript for video ID {video_id}: No transcript found.")
#         return None
#     except TranscriptsDisabled:
#         logging.warning(f"Could not fetch transcript for video ID {video_id}: Subtitles are disabled.")
#         return None
#     except Exception as e:
#         logging.error(f"An error occurred while fetching transcript for video ID {video_id}: {e}")
#         return None

# # List of domains (queries)
# domains = ["Technology", "Education", "Entertainment", "Gaming", "Lifestyle", "Vlogs", 
#            "News and Politics", "Music", "Fitness and Health", "Cooking and Food", 
#            "Beauty and Makeup", "Science", "Sports", "Travel", "Business and Finance", 
#            "Animation and Film Making", "Crafts and DIY", "Motivational", "History", 
#            "Animals and Pets"]

# # Collect data
# data = []
# for domain in domains:
#     logging.info(f"Collecting data for domain: {domain}")
#     video_details = get_video_details(domain, max_results=50)
#     for item in video_details['items']:
#         video_id = item['id']['videoId']
#         title = item['snippet']['title']
#         description = item['snippet']['description']
#         transcript = get_video_transcript(video_id)
#         if transcript:
#             data.append((title + " " + description + " " + transcript, domain))
#         else:
#             logging.info(f"Skipping video ID {video_id} due to unavailable transcript.")

# # Create DataFrame
# df = pd.DataFrame(data, columns=['text', 'label'])
# df.to_csv('large_dataset.csv', index=False)

# logging.info("Data collection completed successfully.")


# from googleapiclient.discovery import build
# import pandas as pd
# from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
# import logging
# import time

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Set up YouTube API
# api_key = ''
# youtube = build('youtube', 'v3', developerKey=api_key)

# # Function to get video transcript
# def get_video_transcript(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         return ' '.join([t['text'] for t in transcript])
#     except NoTranscriptFound:
#         logging.warning(f"Could not fetch transcript for video ID {video_id}: No transcript found.")
#         return None
#     except TranscriptsDisabled:
#         logging.warning(f"Could not fetch transcript for video ID {video_id}: Subtitles are disabled.")
#         return None
#     except Exception as e:
#         logging.error(f"An error occurred while fetching transcript for video ID {video_id}: {e}")
#         return None

# # List of domains (queries)
# domains = ["Technology", "Education", "Entertainment", "Gaming", "Lifestyle", "Vlogs",
#            "News and Politics", "Music", "Fitness and Health", "Cooking and Food",
#            "Beauty and Makeup", "Science", "Sports", "Travel", "Business and Finance",
#            "Animation and Film Making", "Crafts and DIY", "Motivational", "History",
#            "Animals and Pets"]

# data = []
# max_results = 50  # Maximum number of results per request
# batch_size = 10   # Number of requests to make per batch

# for domain in domains:
#     query = domain
#     next_page_token = None
#     while True:
#         request = youtube.search().list(
#             #q=query,
#             part='snippet',
#             # type='video',
#             maxResults=max_results,
#             videoCategoryId='1',
#             # chart='mostPopular',
#             regionCode="IN",
#             # pageToken=next_page_token
#         )
#         response = request.execute()

#         for item in response['items']:
#             video_id = item['id']
#             title = item['snippet']['title']
#             description = item['snippet']['description']
#             transcript = get_video_transcript(video_id)
#             if transcript:
#                 data.append((f"{title} {description} {transcript}", domain))
#             else:
#                 logging.info(f"Skipping video ID {video_id} due to unavailable transcript.")

#         next_page_token = response.get('nextPageToken')
#         if not next_page_token:
#             break

#         # Introduce a delay between batches of requests
#         if len(data) % (batch_size * max_results) == 0:
#             time.sleep(60)  # Wait for 60 seconds before making the next batch of requests

# df = pd.DataFrame(data, columns=['text', 'label'])
# df.to_csv('dataset.csv', index=False)
# logging.info("Data collection completed successfully.")

# YouTube video category IDs mapped to your domains
import pandas as pd
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# YouTube API setup
api_key = ''
youtube = build('youtube', 'v3', developerKey=api_key)

# Define YouTube category IDs mapped to your domains
category_to_domain = {
    '1': 'Film & Animation',
    '2': 'Autos & Vehicles',
    '10': 'Music',
    '15': 'Pets & Animals',
    '17': 'Sports',
    '19': 'Travel & Events',
    '20': 'Gaming',
    '22': 'People & Blogs',
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Howto & Style',
    '27': 'Education',
    '28': 'Science & Technology',
    '29': 'Nonprofits & Activism'
}

# Define domains as per the above mapping
domains = list(category_to_domain.values())

# Function to fetch videos for a given category ID
def fetch_videos_for_category(category_id, max_results=10):
    request = youtube.search().list(
        part="snippet",
        type="video",
        videoCategoryId=category_id,
        maxResults=max_results
    )
    response = request.execute()
    return response

# Function to get the transcript of a video
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        print(f"Could not fetch transcript for video ID {video_id}: {e}")
        return None

# Fetch videos and their transcripts
videos = []
for category_id in category_to_domain.keys():
    response = fetch_videos_for_category(category_id)
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        transcript = get_transcript(video_id)
        if transcript:
            video_info = {
                'videoId': video_id,
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'transcript': transcript,
                'domain': category_to_domain[category_id]
            }
            videos.append(video_info)

# Create a DataFrame
df = pd.DataFrame(videos)
df.to_csv('dataset.csv', index=False)
# Preprocess the text and labels
# df['text'] = df['title'] + " " + df['description'] + " " + df['transcript']

# # Map domain labels to numeric labels
# domain_to_id = {domain: idx for idx, domain in enumerate(domains)}
# df['label'] = df['domain'].apply(lambda x: domain_to_id[x])

# # Split the dataset
# train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'])

# # Load tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(domains))

# # Tokenize the data
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding='max_length', truncation=True)

# train_dataset = Dataset.from_pandas(train_df)
# val_dataset = Dataset.from_pandas(val_df)

# train_dataset = train_dataset.map(tokenize_function, batched=True)
# val_dataset = val_dataset.map(tokenize_function, batched=True)

# train_dataset = train_dataset.rename_column("label", "labels")
# val_dataset = val_dataset.rename_column("label", "labels")

# # Set training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train the model
# trainer.train()

# # Evaluate the model
# results = trainer.evaluate()

# print(results)
