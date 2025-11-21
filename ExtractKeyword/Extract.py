import requests
import pandas as pd
import time
import re
from collections import Counter
from datetime import datetime

# ---------- CONFIGURATION ----------

# Subreddits to monitor
subreddits = [
    "SkincareAddiction", "Acne", "AcneBust", "eczema", "psoriasis",
    "rosacea", "Tretinoin", "AsianBeauty", "30PlusSkinCare"
]

posts_per_sub = 100

# API Keys (Replace with your actual keys)
API_KEYS = {
    'instagram': 'YOUR_INSTAGRAM_API_KEY',
    'tiktok': 'YOUR_TIKTOK_API_KEY',
    'youtube': 'YOUR_YOUTUBE_API_KEY',
    'twitter': 'YOUR_TWITTER_BEARER_TOKEN',
    'spotify': 'YOUR_SPOTIFY_API_KEY'  # For podcasts
}

# ---------- 1. REDDIT SCRAPER ----------

def fetch_subreddit_new(subreddit, limit=100):
    """Fetch posts from a subreddit"""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {'User-Agent': 'Hackathon-Eucerin-VoiceOfSkin/0.1 by Shifa'}
    
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching r/{subreddit}: {e}")
        return pd.DataFrame()
    
    data = resp.json()
    if not data.get('data', {}).get('children'):
        print(f"No posts returned for r/{subreddit}")
        return pd.DataFrame()
    
    records = []
    for post in data['data']['children']:
        post_data = post['data']
        records.append({
            'platform': 'reddit',
            'source': f"r/{subreddit}",
            'post_id': post_data.get('id', ''),
            'timestamp': pd.to_datetime(post_data.get('created_utc', 0), unit='s', utc=True),
            'text_content': post_data.get('selftext', ''),
            'title': post_data.get('title', ''),
            'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
            'engagement_metric': post_data.get('score', 0),
            'author': post_data.get('author', ''),
            'comments_count': post_data.get('num_comments', 0)
        })
    
    return pd.DataFrame(records)

def fetch_reddit_data(subreddits_list):
    """Fetch data from multiple subreddits"""
    all_dfs = []
    for sub in subreddits_list:
        print(f"Fetching r/{sub} ...")
        df_sub = fetch_subreddit_new(sub, limit=posts_per_sub)
        if not df_sub.empty:
            all_dfs.append(df_sub)
        time.sleep(1)
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ---------- 2. INSTAGRAM SCRAPER ----------

def fetch_instagram_posts(hashtags, access_token):
    """
    Fetch Instagram posts by hashtags
    Note: Requires Instagram Graph API access
    """
    records = []
    
    for hashtag in hashtags:
        url = f"https://graph.instagram.com/ig_hashtag_search"
        params = {
            'user_id': 'YOUR_USER_ID',
            'q': hashtag,
            'access_token': access_token
        }
        
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            # Get recent media for hashtag
            if data.get('data'):
                hashtag_id = data['data'][0]['id']
                media_url = f"https://graph.instagram.com/{hashtag_id}/recent_media"
                media_params = {
                    'user_id': 'YOUR_USER_ID',
                    'fields': 'id,caption,timestamp,like_count,comments_count,media_url,permalink',
                    'access_token': access_token
                }
                
                media_resp = requests.get(media_url, params=media_params)
                media_data = media_resp.json()
                
                for post in media_data.get('data', []):
                    records.append({
                        'platform': 'instagram',
                        'source': f"#{hashtag}",
                        'post_id': post.get('id', ''),
                        'timestamp': pd.to_datetime(post.get('timestamp', '')),
                        'text_content': post.get('caption', ''),
                        'title': '',
                        'url': post.get('permalink', ''),
                        'engagement_metric': post.get('like_count', 0),
                        'author': '',
                        'comments_count': post.get('comments_count', 0)
                    })
        except Exception as e:
            print(f"Error fetching Instagram #{hashtag}: {e}")
        
        time.sleep(2)
    
    return pd.DataFrame(records)

# ---------- 3. TIKTOK SCRAPER ----------

def fetch_tiktok_videos(hashtags, api_key):
    """
    Fetch TikTok videos by hashtags
    Note: Requires TikTok API access
    """
    records = []
    
    for hashtag in hashtags:
        url = "https://open.tiktokapis.com/v2/research/video/query/"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "query": {
                "and": [{"field_name": "hashtag_name", "field_values": [hashtag]}]
            },
            "max_count": 100
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            for video in data.get('data', {}).get('videos', []):
                records.append({
                    'platform': 'tiktok',
                    'source': f"#{hashtag}",
                    'post_id': video.get('id', ''),
                    'timestamp': pd.to_datetime(video.get('create_time', 0), unit='s'),
                    'text_content': video.get('video_description', ''),
                    'title': '',
                    'url': f"https://www.tiktok.com/@{video.get('username', '')}/video/{video.get('id', '')}",
                    'engagement_metric': video.get('like_count', 0),
                    'author': video.get('username', ''),
                    'comments_count': video.get('comment_count', 0)
                })
        except Exception as e:
            print(f"Error fetching TikTok #{hashtag}: {e}")
        
        time.sleep(2)
    
    return pd.DataFrame(records)

# ---------- 4. TWITTER/X SCRAPER ----------

def fetch_twitter_posts(keywords, bearer_token):
    """
    Fetch tweets by keywords
    Note: Requires Twitter API v2 access
    """
    records = []
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {'Authorization': f'Bearer {bearer_token}'}
    
    for keyword in keywords:
        params = {
            'query': keyword,
            'max_results': 100,
            'tweet.fields': 'created_at,public_metrics,author_id',
            'expansions': 'author_id'
        }
        
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            for tweet in data.get('data', []):
                metrics = tweet.get('public_metrics', {})
                records.append({
                    'platform': 'twitter',
                    'source': f'keyword:{keyword}',
                    'post_id': tweet.get('id', ''),
                    'timestamp': pd.to_datetime(tweet.get('created_at', '')),
                    'text_content': tweet.get('text', ''),
                    'title': '',
                    'url': f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                    'engagement_metric': metrics.get('like_count', 0),
                    'author': tweet.get('author_id', ''),
                    'comments_count': metrics.get('reply_count', 0)
                })
        except Exception as e:
            print(f"Error fetching Twitter '{keyword}': {e}")
        
        time.sleep(2)
    
    return pd.DataFrame(records)

# ---------- 5. YOUTUBE SCRAPER ----------

def fetch_youtube_videos(keywords, api_key):
    """
    Fetch YouTube videos by keywords
    """
    records = []
    url = "https://www.googleapis.com/youtube/v3/search"
    
    for keyword in keywords:
        params = {
            'part': 'snippet',
            'q': keyword,
            'type': 'video',
            'maxResults': 50,
            'key': api_key,
            'order': 'date'
        }
        
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            for item in data.get('items', []):
                snippet = item.get('snippet', {})
                video_id = item.get('id', {}).get('videoId', '')
                
                # Get video statistics
                stats_url = "https://www.googleapis.com/youtube/v3/videos"
                stats_params = {
                    'part': 'statistics',
                    'id': video_id,
                    'key': api_key
                }
                stats_resp = requests.get(stats_url, params=stats_params)
                stats_data = stats_resp.json()
                statistics = stats_data.get('items', [{}])[0].get('statistics', {})
                
                records.append({
                    'platform': 'youtube',
                    'source': f'search:{keyword}',
                    'post_id': video_id,
                    'timestamp': pd.to_datetime(snippet.get('publishedAt', '')),
                    'text_content': snippet.get('description', ''),
                    'title': snippet.get('title', ''),
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'engagement_metric': int(statistics.get('likeCount', 0)),
                    'author': snippet.get('channelTitle', ''),
                    'comments_count': int(statistics.get('commentCount', 0))
                })
        except Exception as e:
            print(f"Error fetching YouTube '{keyword}': {e}")
        
        time.sleep(2)
    
    return pd.DataFrame(records)

# ---------- 6. PODCAST SCRAPER (Spotify) ----------

def fetch_spotify_podcasts(keywords, api_key):
    """
    Fetch podcast episodes from Spotify
    Note: Requires Spotify API access
    """
    records = []
    
    # First, get access token
    auth_url = "https://accounts.spotify.com/api/token"
    auth_data = {'grant_type': 'client_credentials'}
    # You'll need to add client_id and client_secret
    
    for keyword in keywords:
        url = "https://api.spotify.com/v1/search"
        headers = {'Authorization': f'Bearer {api_key}'}
        params = {
            'q': keyword,
            'type': 'episode',
            'limit': 50
        }
        
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            for episode in data.get('episodes', {}).get('items', []):
                records.append({
                    'platform': 'podcast_spotify',
                    'source': episode.get('show', {}).get('name', ''),
                    'post_id': episode.get('id', ''),
                    'timestamp': pd.to_datetime(episode.get('release_date', '')),
                    'text_content': episode.get('description', ''),
                    'title': episode.get('name', ''),
                    'url': episode.get('external_urls', {}).get('spotify', ''),
                    'engagement_metric': 0,  # Spotify doesn't provide like counts
                    'author': episode.get('show', {}).get('publisher', ''),
                    'comments_count': 0
                })
        except Exception as e:
            print(f"Error fetching Spotify podcasts '{keyword}': {e}")
        
        time.sleep(2)
    
    return pd.DataFrame(records)

# ---------- 7. MAIN DATA COLLECTION ----------

def collect_all_data():
    """Collect data from all platforms"""
    
    # Define search terms
    skin_search_terms = [
        "skincare", "eczema", "psoriasis", "acne", "rosacea",
        "dermatitis", "skin barrier", "sensitive skin"
    ]
    
    all_data = []
    
    # Reddit
    print("\n--- Fetching Reddit Data ---")
    reddit_data = fetch_reddit_data(subreddits)
    if not reddit_data.empty:
        all_data.append(reddit_data)
        print(f"Reddit: {len(reddit_data)} posts collected")
    
    # Instagram (uncomment when you have API access)
    # print("\n--- Fetching Instagram Data ---")
    # instagram_data = fetch_instagram_posts(skin_search_terms, API_KEYS['instagram'])
    # if not instagram_data.empty:
    #     all_data.append(instagram_data)
    #     print(f"Instagram: {len(instagram_data)} posts collected")
    
    # TikTok (uncomment when you have API access)
    # print("\n--- Fetching TikTok Data ---")
    # tiktok_data = fetch_tiktok_videos(skin_search_terms, API_KEYS['tiktok'])
    # if not tiktok_data.empty:
    #     all_data.append(tiktok_data)
    #     print(f"TikTok: {len(tiktok_data)} posts collected")
    
    # Twitter (uncomment when you have API access)
    # print("\n--- Fetching Twitter Data ---")
    # twitter_data = fetch_twitter_posts(skin_search_terms, API_KEYS['twitter'])
    # if not twitter_data.empty:
    #     all_data.append(twitter_data)
    #     print(f"Twitter: {len(twitter_data)} posts collected")
    
    # YouTube (uncomment when you have API access)
    # print("\n--- Fetching YouTube Data ---")
    # youtube_data = fetch_youtube_videos(skin_search_terms, API_KEYS['youtube'])
    # if not youtube_data.empty:
    #     all_data.append(youtube_data)
    #     print(f"YouTube: {len(youtube_data)} posts collected")
    
    # Spotify Podcasts (uncomment when you have API access)
    # print("\n--- Fetching Podcast Data ---")
    # podcast_data = fetch_spotify_podcasts(skin_search_terms, API_KEYS['spotify'])
    # if not podcast_data.empty:
    #     all_data.append(podcast_data)
    #     print(f"Podcasts: {len(podcast_data)} posts collected")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n--- Total posts collected: {len(combined_df)} ---")
        return combined_df
    else:
        print("No data collected")
        return pd.DataFrame()

# ---------- 8. TEXT CLEANING AND ANALYSIS ----------

# Stop words
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

custom_stopwords = {
    "skin", "amp", "ve", "don", "feel", "product", "products",
    "2", "im", "ive", "id", "didnt", "3", "1", "bad",
    "month", "months", "started", "start", "week",
    "day", "days", "help", "get", "got",
    "really", "does", "use", "using", "used",
    "just", "like", "know", "think", "one",
    "also", "bit", "i'm", "make", "made"
}

all_stopwords = stop_words | custom_stopwords

skin_keywords = [
    "skin", "eczema", "psoriasis", "rash", "rashes", "itch", "itchy", "eucerin", "skin care",
    "flare", "flare-up", "flareups", "acne", "pimples", "blackheads",
    "whiteheads", "rosacea", "dermatitis", "hyperpigmentation", "dark spots",
    "redness", "sensitive", "barrier", "skin barrier", "spf", "sunscreen",
    "dryness", "dry skin", "oily skin", "combination skin", "scarring",
    "acne scars", "maskne", "allergic reaction", "hives", "bumps"
]

skin_pattern = '|'.join(skin_keywords)

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def contains_skin_keyword(text):
    if not text:
        return False
    return bool(re.search(skin_pattern, text, re.IGNORECASE))

def analyze_ngrams(df, top_k=50):
    """Analyze unigrams and bigrams from text data"""
    
    # Create full text
    df['full_text'] = df.apply(
        lambda row: f"{row['title']} {row['text_content']}" if len(str(row['text_content'])) > 0 else str(row['title']),
        axis=1
    )
    df['clean_text'] = df['full_text'].apply(clean_text)
    df['is_skin'] = df['clean_text'].apply(contains_skin_keyword)
    
    # Filter for skin-related content
    df_filtered = df[df['is_skin'] & (df['clean_text'].str.len() > 0)].copy()
    
    print(f"Filtered to {len(df_filtered)} skin-related posts")
    
    # Unigrams
    words = []
    for text in df_filtered['clean_text']:
        words.extend(text.split())
    
    words_filtered = [w for w in words if w not in all_stopwords and len(w) > 1]
    unigram_counts = Counter(words_filtered)
    
    unigrams = pd.DataFrame([
        {'ngram': word, 'count': count, 'type': 'unigram'}
        for word, count in unigram_counts.items()
    ])
    
    # Bigrams
    def generate_bigrams(text):
        words = text.split()
        return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    bigrams_list = []
    for text in df_filtered['clean_text']:
        bigrams_list.extend(generate_bigrams(text))
    
    bigram_counts = Counter(bigrams_list)
    
    bigrams_filtered_list = []
    for bigram, count in bigram_counts.items():
        parts = bigram.split()
        if len(parts) == 2:
            w1, w2 = parts
            if w1 not in all_stopwords and w2 not in all_stopwords:
                bigrams_filtered_list.append({'ngram': bigram, 'count': count, 'type': 'bigram'})
    
    bigrams_df = pd.DataFrame(bigrams_filtered_list)
    
    # Combine and get top k
    ngrams_top = pd.concat([unigrams, bigrams_df], ignore_index=True)
    ngrams_top = ngrams_top.sort_values('count', ascending=False).head(top_k)
    
    total_count = ngrams_top['count'].sum()
    ngrams_top['percent'] = round((ngrams_top['count'] / total_count) * 100, 2)
    
    return df_filtered, ngrams_top

# ---------- 9. RUN THE ANALYSIS ----------

if __name__ == "__main__":
    # Collect data from all platforms
    all_data = collect_all_data()
    
    if not all_data.empty:
        # Analyze the data
        filtered_data, ngrams_results = analyze_ngrams(all_data)
        
        # Display results
        print("\n--- Top N-grams ---")
        print(ngrams_results.head(20))
        
        # Platform breakdown
        print("\n--- Posts by Platform ---")
        print(filtered_data['platform'].value_counts())
        
        # Save results
        filtered_data.to_csv('skin_social_media_data.csv', index=False)
        ngrams_results.to_csv('skin_ngrams_analysis.csv', index=False)
        print("\nData saved to CSV files")