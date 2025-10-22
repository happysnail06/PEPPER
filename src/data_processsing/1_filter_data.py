"""
    - This code aligns the IMDB user data with ReDial & OpenDialKG dataset.
    - It identifies common movies between the user dataset and each CRS dataset, and removes uncommon entries from each user's profile. 
    - Then, it filters out users with less than {CORE} interaction histories. 
"""

import json
import re
import os


### Minimum User Interaction History ###
core = 10

### File Paths ###
USER_DATA_PATH = 'dataset/user_data/user_data_IMDB.json'

OPENDIALKG_PATH = 'crs_data/opendialkg/id2info.json'
REDIAL_PATH = 'crs_data/redial/id2info.json'

# Add paths for movie plot data
OPENDIALKG_MOVIE_DATA_PATH = 'dataset/movie_data/opendialkg_movie_data.json'
REDIAL_MOVIE_DATA_PATH = 'dataset/movie_data/redial_movie_data.json'

OPENDIALKG_OUTPUT_DIR = 'dataset/user_data/opendialkg'
REDIAL_OUTPUT_DIR = 'dataset/user_data/redial'

os.makedirs(OPENDIALKG_OUTPUT_DIR, exist_ok=True)
os.makedirs(REDIAL_OUTPUT_DIR, exist_ok=True)


def load_user_data(file_path):
    """Load and clean user data from a given JSON file."""
    with open(file_path) as file:
        raw_data = json.load(file)
    return {user: reviews for user, reviews in raw_data.items() if reviews}

def process_movies(movie_list, remove_year=True):
    """Process movie titles by optionally removing years."""
    if remove_year:
        return set(re.sub(r'\s*\(\d{4}\)', '', movie) for movie in movie_list)
    return set(movie_list)

def load_dataset(file_path, remove_year=True):
    """Load a dataset and process movie titles."""
    with open(file_path) as file:
        data = json.load(file)
    return process_movies([info["name"] for info in data.values()], remove_year=remove_year)

def save_filtered_data(directory, file_name, data):
    """Save filtered data to a JSON file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    with open(f'{directory}/{file_name}', 'w') as file:
        json.dump(data, file, indent=4)

def filter_user_data_opendialkg(data, common_movies, core, movie_plot_data):
    """Filter user data for OpenDialKG, removing years, adding plots, renaming keys, and ordering fields."""
    filtered_users = {}
    for user_id, movie_entries in data.items():
        processed_movies = []
        for movie in movie_entries:
            title_no_year = re.sub(r'\s*\(\d{4}\)', '', movie["title"])
            if title_no_year in common_movies:
                # Prepare data, including renaming
                original_title = movie.get("title") # Keep original title if needed, but use title_no_year for keying/output
                plot_content = movie_plot_data.get(title_no_year, {}).get("plot")
                review_raw_content = movie.get("raw_review")
                review_abstract_content = movie.get("abstract")

                # Construct the movie entry with specified order and renamed keys
                ordered_movie = {
                    "user_id": user_id, # Add user_id here
                    "imdb_id": movie.get("imdb_id"),
                    "title": title_no_year, # Use title without year
                    "rating": movie.get("rating"),
                    "genres": movie.get("genres"),
                    "director": movie.get("director"),
                    "cast": movie.get("cast"),
                    "date": movie.get("date"),
                    "plot": plot_content,
                    "review_raw": review_raw_content,
                    "review_abstract": review_abstract_content,
                }
                # Remove keys with None values if desired (optional, keeps structure clean)
                ordered_movie = {k: v for k, v in ordered_movie.items() if v is not None}

                processed_movies.append(ordered_movie)

        if len(processed_movies) >= core:
            filtered_users[user_id] = processed_movies
    return filtered_users

def filter_user_data_redial(data, common_movies, core, movie_plot_data):
    """Filter user data for ReDial, adding plots, renaming keys, and ordering fields."""
    filtered_users = {}
    for user_id, movie_entries in data.items():
        processed_movies = []
        for movie in movie_entries:
            title = movie["title"]
            if title in common_movies:
                 # Prepare data, including renaming
                plot_content = movie_plot_data.get(title, {}).get("plot")
                review_raw_content = movie.get("raw_review")
                review_abstract_content = movie.get("abstract")

                # Construct the movie entry with specified order and renamed keys
                ordered_movie = {
                    "user_id": user_id, # Add user_id here
                    "imdb_id": movie.get("imdb_id"),
                    "title": title, # Keep original title
                    "genres": movie.get("genres"),
                    "director": movie.get("director"),
                    "cast": movie.get("cast"),
                    "date": movie.get("date"),
                    "rating": movie.get("rating"),
                    "plot": plot_content,
                    "review_raw": review_raw_content,
                    "review_abstract": review_abstract_content,
                }
                # Remove keys with None values if desired (optional, keeps structure clean)
                ordered_movie = {k: v for k, v in ordered_movie.items() if v is not None}

                processed_movies.append(ordered_movie)

        if len(processed_movies) >= core:
            filtered_users[user_id] = processed_movies
    return filtered_users

def process_opendialkg(user_data):
    """Process the OpenDialKG dataset."""
    print("\n# OpenDialKG")
    
    imdb_movies = process_movies(
        [review['title'] for reviews in user_data.values() for review in reviews]
    )
    print(f"IMDB movie count, year removed: {len(imdb_movies)}")
    
    opendialkg_movies = load_dataset(OPENDIALKG_PATH)
    print(f"OpenDialKG movie count: {len(opendialkg_movies)}")
    
    common_movies = imdb_movies.intersection(opendialkg_movies)
    print(f"Number of common movies in IMDB & OpenDialKG: {len(common_movies)}")
    
    # Load OpenDialKG movie plot data
    with open(OPENDIALKG_MOVIE_DATA_PATH) as file:
        opendialkg_plot_data = json.load(file)
    print(f"Loaded plot data for {len(opendialkg_plot_data)} OpenDialKG movies.")

    filtered_users = filter_user_data_opendialkg(user_data, common_movies, core, opendialkg_plot_data)
    print(f"Number of users with at least {core} movies: {len(filtered_users)}")
    
    save_filtered_data(OPENDIALKG_OUTPUT_DIR, f'test_opendialkg_{core}_filtered.json', filtered_users)

def process_redial(user_data):
    """Process the ReDial dataset."""
    print("\n# ReDial")
    
    imdb_movies = process_movies(
        [review['title'] for reviews in user_data.values() for review in reviews], remove_year=False
    )
    print(f"IMDB movie count: {len(imdb_movies)}")
    
    redial_movies = load_dataset(REDIAL_PATH, remove_year=False)
    print(f"ReDial movie count: {len(redial_movies)}")
    
    common_movies = imdb_movies.intersection(redial_movies)
    print(f"Number of common movies in IMDB & ReDial: {len(common_movies)}")
    
    # Load ReDial movie plot data
    with open(REDIAL_MOVIE_DATA_PATH) as file:
        redial_plot_data = json.load(file)
    print(f"Loaded plot data for {len(redial_plot_data)} ReDial movies.")

    filtered_users = filter_user_data_redial(user_data, common_movies, core, redial_plot_data)
    print(f"Number of users with at least {core} movies: {len(filtered_users)}")
    
    save_filtered_data(REDIAL_OUTPUT_DIR, f'test_redial_{core}_filtered.json', filtered_users)

if __name__ == "__main__":
    user_data = load_user_data(USER_DATA_PATH)
    
    process_opendialkg(user_data)
    process_redial(user_data)