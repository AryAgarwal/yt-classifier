# yt-classifier
A python app that takes in a youtube video link and classifies it as one of the following domains: Technology, Education, Entertainment, Gaming, Lifestyle, Vlogs, News and Politics, Music, Fitness and Health, Cooking and Food, Beauty and Makeup, Science, Sports, Travel, Business and Finance, Animation and Film Making, Crafts and DIY, Motivational, History, Animals and Pets.

Created using FastAPI and a front-end HTMl page. Video id, title, transcription and classification stored in videos.db. Used whisper to transcribe the downloaded audio files into text and then pass it onto DistilBert model from HuggingFace for classification.

Model was not giving accurate results so decided to create a custom dataset using the Youtube API to fetch videos and their transciptions. However, that feature is not currently working.

In the colab notebook, I have fine-tuned the model using a dataset from kaggle. But the dataset is not very helpful so predictions could still be more accurate.

To run the project, run index.html on a live server and keep the backend running by typing python main.py
