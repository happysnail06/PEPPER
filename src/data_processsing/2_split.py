"""
    - This code split the user interaction histories to SEEN and TARGET.
"""

import json
import numpy as np
import random
import math

# Global Variables
CORE = 10
DATASETS = ["redial", "opendialkg"]

def load_data(path):
    """
    Load JSON data from the given path.
    """
    with open(path, 'r') as file:
        return json.load(file)

def save_data(path, data):
    """
    Save JSON data to the given path.
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def extract_likes_dislikes(abstract):
    """
    Extract likes and dislikes from the abstract text.
    Remove "[Like]" and "[Dislike]" tags.
    """
    likes = []
    dislikes = []
    
    like_tag = "[Like]"
    dislike_tag = "[Dislike]"
    
    like_start_idx = abstract.find(like_tag)
    dislike_start_idx = abstract.find(dislike_tag)

    if like_start_idx != -1:
        # Determine the end of the likes section
        # It ends where [Dislike] starts, or at the end of the abstract if [Dislike] isn't after [Like]
        end_of_likes_idx = len(abstract)
        if dislike_start_idx != -1 and dislike_start_idx > like_start_idx:
            end_of_likes_idx = dislike_start_idx
        
        likes_content = abstract[like_start_idx + len(like_tag):end_of_likes_idx].strip()
        if "None" not in likes_content and likes_content: # ensure not empty string
            likes = [line.strip() for line in likes_content.splitlines() if line.strip()]

    if dislike_start_idx != -1:
        # Dislikes section starts after [Dislike] tag and goes to the end of the abstract
        # (or end of relevant section if there were other tags, but not specified here)
        dislikes_content = abstract[dislike_start_idx + len(dislike_tag):].strip()
        if "None" not in dislikes_content and dislikes_content: # ensure not empty string
            dislikes = [line.strip() for line in dislikes_content.splitlines() if line.strip()]

    return likes, dislikes

def process_user_data(data):
    """
    Process data for each user: sort by rating, split into 'seen' and 'target', 
    extract likes, dislikes, and calculate average rating from seen items.
    """
    processed_data = {}

    def get_numeric_rating(review):
        """Helper to safely extract numeric rating for sorting."""
        try:
            # Assumes rating is like '8/10' or just '8'. Takes the part before '/'.
            rating_str = review.get('rating', '0').split('/')[0]
            return int(rating_str)
        except (ValueError, AttributeError):
            return 0 # Default to 0 for missing, malformed, or non-numeric ratings

    for user_id, reviews in data.items():
        if not reviews: 
            processed_data[user_id] = {
                'average rating': 0.0, 'seen/target': '0 / 0', 'seen movies': [], 
                'seen genres': [], 'target movies': [], 'likes': [], 'dislikes': [],
                'seen': [], 'target': []
            }
            continue

        # Sort reviews by rating (highest first)
        sorted_reviews = sorted(reviews, key=get_numeric_rating, reverse=True)
        
        num_total_reviews = len(sorted_reviews)
        
        # Since num_total_reviews will be at least 10 from the previous step (CORE_THRESHOLD = 10)
        # we can directly apply the 80/20 split.
        num_seen_items = math.ceil(num_total_reviews * 0.8)
        num_target_items = num_total_reviews - num_seen_items

        target_reviews = sorted_reviews[:num_target_items] # Highest rated go to target
        seen_reviews = sorted_reviews[num_target_items:]   # The rest go to seen

        seen_movies = [review['title'] for review in seen_reviews]
        seen_genres = list(set(genre for review in seen_reviews for genre in review.get('genres', [])))
        target_movies = [review['title'] for review in target_reviews]

        all_likes, all_dislikes = [], []
        avg_rating_val = 0.0
        
        if seen_reviews: 
            ratings_values = [get_numeric_rating(review) for review in seen_reviews]
            if ratings_values:
                avg_rating_val = np.mean(ratings_values)

            for review in seen_reviews:
                likes, dislikes = extract_likes_dislikes(review.get("review_abstract", ""))
                all_likes.extend(likes)
                all_dislikes.extend(dislikes)
        
        processed_data[user_id] = {
            'average rating': round(avg_rating_val, 2),
            'seen/target': f'{len(seen_reviews)} / {len(target_reviews)}',
            'seen movies': seen_movies,
            'seen genres': seen_genres,
            'target movies': target_movies,
            'likes': all_likes,
            'dislikes': all_dislikes,
            'seen': seen_reviews,
            'target': target_reviews
        }

    return processed_data


if __name__ == "__main__":
    
    for dataset in DATASETS:
        DATA_PATH = f'dataset/user_data/{dataset}/1_{dataset}_{CORE}_filtered.json'
        SAVE_DIR = f'dataset/user_data/{dataset}/2_{dataset}_{CORE}_split.json'
        
        data = load_data(DATA_PATH)
        print(f"Loaded {len(data)} users from {dataset}.")
        
        processed_data = process_user_data(data)
        
        save_data(SAVE_DIR, processed_data)
        print(f"Processed data for {dataset} saved to: {SAVE_DIR}")
