# METHOD 1- DISTILBERT

import os
import torch
import asyncio
import yt_dlp
import whisper
import sqlite3
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# Set up FastAPI app
app = FastAPI()

# Define the list of domains
domains = [
    "Technology", "Education", "Entertainment", "Gaming", "Lifestyle", "Vlogs",
    "News and Politics", "Music", "Fitness and Health", "Cooking and Food",
    "Beauty and Makeup", "Science", "Sports", "Travel", "Business and Finance",
    "Animation and Film Making", "Crafts and DIY", "Motivational", "History",
    "Animals and Pets"
]

# Set up SQLite database
conn = sqlite3.connect('videos.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS videos
             (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, transcript TEXT, classification TEXT, justification TEXT, metadata TEXT)''')
conn.commit()

# Load the pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(domains))

# Function to download and extract audio from a YouTube video
async def download_and_extract_audio(video_url):
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

    return 'audio.mp3'

# Function to transcribe audio using Whisper API
async def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# Function to classify transcription using the fine-tuned DistilBERT model
def classify_transcription(transcript):
    inputs = tokenizer(transcript, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)[0]
    predicted_class_idx = torch.argmax(outputs).item()
    predicted_class = domains[predicted_class_idx]
    return [{'domain': predicted_class, 'justification': f'The transcription is classified as {predicted_class}'}]

# Asynchronous function to process a video
async def process_video(video_url, background_tasks):
    # Download and extract audio
    audio_file = await download_and_extract_audio(video_url)

    # Transcribe audio
    transcript = await transcribe_audio(audio_file)

    # Classify transcription
    classifications = classify_transcription(transcript)

    # Store video metadata, transcript, and classifications in the database
    metadata = f"URL: {video_url}"
    c.execute("INSERT INTO videos (url, transcript, classification, justification, metadata) VALUES (?, ?, ?, ?, ?)",
              (video_url, transcript, str(classifications), classifications[0]['justification'], metadata))
    conn.commit()

    # Clean up the audio file
    os.remove(audio_file)

    return {"message": "Video processing completed"}

# API endpoint to process a video
class VideoURL(BaseModel):
    url: str

@app.post("/process_video")
async def process_video_endpoint(video_url: VideoURL, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video, video_url.url, background_tasks)
    return {"message": "Video processing started"}

# API endpoint to retrieve processed videos
@app.get("/videos")
def get_videos():
    c.execute("SELECT * FROM videos")
    videos = c.fetchall()
    return [{'id': video[0], 'url': video[1], 'transcript': video[2], 'classification': video[3], 'justification': video[4], 'metadata': video[5]} for video in videos]

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",  # Add this line if your frontend is running on port 8001
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Main function to run the server
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

# METHOD 2 - YOUTUBE MODEL , TF


# import os
# import asyncio
# import yt_dlp
# import whisper
# import sqlite3
# import numpy as np
# import tensorflow as tf
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# # Set up FastAPI app
# app = FastAPI()

# # Set up TensorFlow logging to suppress unnecessary warnings
# tf.get_logger().setLevel('ERROR')

# # Define the list of YouTube video categories
# # These categories are based on the YouTube-8M dataset
# categories = [
#     "Film & Animation", "Autos & Vehicles", "Music", "Pets & Animals",
#     "Sports", "Travel & Events", "Gaming", "People & Blogs",
#     "Comedy", "Entertainment", "News & Politics", "Howto & Style",
#     "Education", "Science & Technology", "Nonprofits & Activism"
# ]

# # Set up SQLite database
# conn = sqlite3.connect('videos.db')
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS videos
#              (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, transcript TEXT, classification TEXT, justification TEXT, metadata TEXT)''')
# conn.commit()

# # Load the pre-trained YouTube-8M model
# model = tf.keras.models.load_model('youtube_8m_model')

# # Function to download and extract audio from a YouTube video
# async def download_and_extract_audio(video_url):
#     # Download video using yt_dlp
#     ydl_opts = {'outtmpl': 'video.mp4'}
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([video_url])

#     # Extract audio using ffmpeg
#     os.system('ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav')

#     return 'audio.wav'

# # Function to transcribe audio using Whisper API
# async def transcribe_audio(audio_file):
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_file)
#     return result["text"]

# # Function to extract features from video using YouTube-8M model
# def extract_video_features(video_path):
#     # Load video using TensorFlow
#     video = tf.io.read_file(video_path)
#     video = tf.io.decode_audio(video, 'wav', desired_channels=2)
#     video = tf.cast(video, tf.float32) / 32767.0  # Normalize audio

#     # Extract audio features using the YouTube-8M model
#     features = model.predict(video)

#     return features

# # Function to classify video category using the YouTube-8M model
# def classify_video_category(features):
#     # Normalize features
#     features = np.array(features) / 255.0
#     # Reshape features to match model input shape
#     features = np.expand_dims(features, axis=0)
#     # Predict video category
#     predictions = model.predict(features)
#     # Get the index of the predicted category
#     predicted_index = np.argmax(predictions)
#     # Get the predicted category label
#     predicted_category = categories[predicted_index]
#     return [{'category': predicted_category, 'justification': f'The video is classified as {predicted_category}'}]

# # Asynchronous function to process a video
# async def process_video(video_url, background_tasks):
#     # Download and extract audio
#     audio_file = await download_and_extract_audio(video_url)

#     # Transcribe audio
#     transcript = await transcribe_audio(audio_file)

#     # Extract features from video
#     features = extract_video_features('video.mp4')

#     # Classify video category
#     classifications = classify_video_category(features)

#     # Store video metadata, transcript, and classifications in the database
#     metadata = f"URL: {video_url}"
#     c.execute("INSERT INTO videos (url, transcript, classification, justification, metadata) VALUES (?, ?, ?, ?, ?)",
#               (video_url, transcript, str(classifications), ', '.join([c['justification'] for c in classifications]), metadata))
#     conn.commit()

#     # Clean up temporary files
#     os.remove(audio_file)
#     os.remove('video.mp4')

#     return {"message": "Video processing completed"}

# # API endpoint to process a video
# class VideoURL(BaseModel):
#     url: str

# @app.post("/process_video")
# async def process_video_endpoint(video_url: VideoURL, background_tasks: BackgroundTasks):
#     background_tasks.add_task(process_video, video_url.url, background_tasks)
#     return {"message": "Video processing started"}

# # API endpoint to retrieve processed videos
# @app.get("/videos")
# def get_videos():
#     c.execute("SELECT * FROM videos")
#     videos = c.fetchall()
#     return [{'id': video[0], 'url': video[1], 'transcript': video[2], 'classification': video[3], 'justification': video[4], 'metadata': video[5]} for video in videos]

# # Add CORS middleware
# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     "http://127.0.0.1:8000",
#     "http://localhost:8001",  # Add this line if your frontend is running on port 8001
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Main function to run the server
# def main():
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     main()
