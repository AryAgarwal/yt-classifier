import os
import pandas as pd
import torch
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
import yt_dlp
import whisper

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
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None

# Function to download and transcribe video using Whisper
def download_and_transcribe_video(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    audio_file = 'audio.mp3'
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    os.remove(audio_file)
    return result["text"]

# Fetch videos and their transcripts
videos = []
for category_id in category_to_domain.keys():
    response = fetch_videos_for_category(category_id)
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        transcript = get_transcript(video_id)
        if not transcript:
            transcript = download_and_transcribe_video(video_id)
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
df.to_csv('videos.csv', index=False)
