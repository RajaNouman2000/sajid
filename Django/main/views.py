from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import re
import os
from django.conf import settings
from django.templatetags.static import static
from django.http import JsonResponse
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
import io
import base64
from collections import Counter
import nltk
from nltk.corpus import stopwords


@csrf_exempt
# Create your views here.
def hashtagworldcloud(request):
    if request.method == 'POST':
        filename = request.FILES['file']
        tweets_df = pd.read_excel(filename)

        # Define the function to clean the text data
        def clean_text(text):
            # Remove hashtags and symbols
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            # Remove emojis
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            # Convert to lowercase
            text = text.lower()
            # Tokenize the text
            tokens = nltk.word_tokenize(text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            stop_words_urdu = set(stopwords.words('urdu'))
            tokens = [
                token for token in tokens if not token in stop_words and not token in stop_words_urdu]
            # Join the tokens back into a string
            text = ' '.join(tokens)
            return text

        # Clean the text data
        tweets_df['cleaned_text'] = tweets_df['Tweet'].apply(clean_text)

        # Count the frequency of keywords in the cleaned text
        keywords = []
        for text in tweets_df['cleaned_text']:
            keywords.extend(text.split())
        keyword_freq = dict(Counter(keywords))

        # Write the keyword frequency data to a text file
        with open('keyword_freq.txt', 'w', encoding='utf-8') as file:
            for keyword, freq in keyword_freq.items():
                file.write(f'{keyword}: {freq}\n')

        # Reshape and reorder the text
        reshaped_text = {}
        for k, v in keyword_freq.items():
            reshaped_k = reshape(k)
            bidi_k = get_display(reshaped_k)
            reshaped_text[bidi_k] = v

        # Generate a word cloud of the hashtags
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10,
                              font_path='main/images/NotoNaskhArabic-Regular.ttf',
                              collocations=False,
                              prefer_horizontal=0.9, max_words=200).generate_from_frequencies(reshaped_text)

        # Plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        # Save the plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Create a response object with the image data
        response = HttpResponse(buffer, content_type='image/png')
        # Set a filename for the downloaded image
        response['Content-Disposition'] = 'attachment; filename="wordcloud.png"'
        # Return the response
        return response


@csrf_exempt
def engagementmetrics(request):
    if request.method == 'POST':
        filename = request.FILES['file']
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(filename)

        max_likes = df.groupby('User')['Likes'].max()
        max_replies = df.groupby('User')['Replies'].max()
        max_retweets = df.groupby('User')['Retweet'].max()

        # Find the user with the tweet with the highest number of likes, replies, and retweets
        max_likes_user = max_likes.idxmax()
        max_replies_user = max_replies.idxmax()
        max_retweets_user = max_retweets.idxmax()

        # Create a bar chart of the number of likes, replies, and retweets for each user
        fig = go.Figure(data=[
            go.Bar(name='Likes', x=max_likes.index, y=max_likes),
            go.Bar(name='Replies', x=max_replies.index, y=max_replies),
            go.Bar(name='Retweets', x=max_retweets.index, y=max_retweets)
        ])

        # Add annotations for the user with the tweet with the highest number of likes, replies, and retweets
        fig.update_layout(
            barmode='group',
            title='User Engagement Metrics',
            xaxis_title='User',
            yaxis_title='Count',
            annotations=[
                dict(x=max_likes_user, y=max_likes[max_likes_user],
                     xref="x", yref="y", text="Max Likes", showarrow=True, arrowhead=1, ax=-30, ay=-40),
                dict(x=max_replies_user, y=max_replies[max_replies_user],
                     xref="x", yref="y", text="Max Replies", showarrow=True, arrowhead=1, ax=-30, ay=-40),
                dict(x=max_retweets_user, y=max_retweets[max_retweets_user],
                     xref="x", yref="y", text="Max Retweets", showarrow=True, arrowhead=1, ax=-30, ay=-40)
            ]
        )

        # Save the plot to a BytesIO object
        buffer = io.BytesIO()
        fig.write_image(buffer, format='png')

        # Set the buffer position to the start
        buffer.seek(0)

        # Create a response object with the image data
        response = HttpResponse(buffer, content_type='image/png')

        # Set a filename for the downloaded image
        response['Content-Disposition'] = 'attachment; filename="plot.png"'

        # Return the response
        return response


@csrf_exempt
def worldcloud(request):
    if request.method == 'POST':
        filename = request.FILES['file']
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(filename)

        # Define function to extract hashtags from each tweet
        def extract_hashtags(text):
            hashtags = re.findall(r'\#\w+', text)
            return hashtags

        # Create a list of all hashtags in the dataframe
        all_hashtags = []
        for tweet in df['Tweet']:
            hashtags = extract_hashtags(tweet)
            all_hashtags.extend(hashtags)

        # Create a dictionary to store the frequency of each hashtag
        frequency = {}
        for hashtag in all_hashtags:
            if hashtag in frequency:
                frequency[hashtag] += 1
            else:
                frequency[hashtag] = 1

        # Write the hashtag frequency to a text file
        with open('hashtag_frequency.txt', 'w', encoding='utf-8') as f:
            for hashtag in frequency:
                f.write('{}\t{}\n'.format(hashtag, frequency[hashtag]))

        # Reshape and reorder the text
        reshaped_text = {}
        for k, v in frequency.items():
            reshaped_k = reshape(k)
            bidi_k = get_display(reshaped_k)
            reshaped_text[bidi_k] = v

        # Generate a word cloud of the hashtags
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10,
                              font_path='main/images/NotoNaskhArabic-Regular.ttf',
                              collocations=False,
                              prefer_horizontal=0.9, max_words=200).generate_from_frequencies(reshaped_text)

        # Plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        image_path = os.path.join(settings.MEDIA_ROOT, 'images/first.png')
        plt.savefig(image_path)
        plt.close()
        with open(image_path, 'rb') as f:
            image_data = f.read()
            # Set the appropriate content type in the response headers
            response = HttpResponse(image_data, content_type='image/png')

            # Optionally, set a filename for the downloaded image
            response['Content-Disposition'] = 'attachment; filename="wordcloud.png"'

            # Return the response
            return response
