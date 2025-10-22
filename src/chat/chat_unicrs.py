import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import asyncio
import aiofiles
import aiohttp  # If needed for other async operations
import tiktoken
import openai
from rapidfuzz import fuzz, process
from tqdm.asyncio import tqdm
import copy

import sys
# Get the absolute path to the 'crs' directory
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
module_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# Add the path to sys.path
sys.path.append(module_path)
sys.path.append(module_path2)

from crs.model.recommender import RECOMMENDER
from src.user_simulator import UserSimulator
from src.utils import *

warnings.filterwarnings('ignore')

SETTING = 'main'
RATIO = 82
TOTAL_COST = 0

def get_entity(text, entity_list):
    extractions = process.extract(text, entity_list, scorer=fuzz.WRatio, limit=20)
    extractions = [extraction[0] for extraction in extractions if extraction[1] >= 90]
    return extractions

def clean_unicrs_response(response: str, candidate_list: list, dataset: str) -> str:
    """
    Clean UNICRS response by replacing special tokens and artifacts with candidate movies.
    
    Args:
        response: Raw response from UNICRS model
        candidate_list: List of candidate movies to substitute
        dataset: Dataset name to determine token types
    
    Returns:
        Cleaned response string
    """
    # Determine movie tokens based on dataset
    if dataset.startswith('redial'):
        movie_tokens = ['<pad>', '<movie>']
    else:
        movie_tokens = ['<mask>', '<pad>']
    
    # Extract response after "System:" and remove endoftext
    if 'System:' in response:
        response = response[response.rfind('System:') + len('System:') + 1:]
    response = response.replace('<|endoftext|>', '')
    
    i = 0
    max_candidates = min(len(candidate_list), 4)
    
    # Replace special tokens with candidates
    for token in movie_tokens:
        token_count = response.count(token)
        for _ in range(token_count):
            if i >= max_candidates:
                break
            response = response.replace(token, candidate_list[i], 1)
            i += 1
    
    # Replace any remaining garbage with candidates
    garbage_patterns = [r'[^\x00-\x7F]{1,3}', r'~{4,}', r'_{4,}']
    for pattern in garbage_patterns:
        if i >= max_candidates:
            break
        while i < max_candidates:
            match = re.search(pattern, response)
            if not match:
                break
            response = response[:match.start()] + candidate_list[i] + response[match.end():]
            i += 1
    
    # Final cleanup: remove artifacts and normalize
    response = re.sub(r'[^\x00-\x7F\s]+', ' ', response)  # Remove non-ASCII
    response = re.sub(r'[,.:;!?]+', lambda m: m.group()[0], response)  # Dedupe punctuation
    response = re.sub(r'(?:\s*[,:;.-]?\s*\d+)+\s*$', '', response)  # Remove trailing numbers
    response = re.sub(r'\s+', ' ', response)  # Normalize spaces
    
    return response.strip()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

async def load_json_async(filepath: str) -> dict:
    async with aiofiles.open(filepath, 'r', encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)

async def save_json_async(filepath: str, data) -> None:
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))

# Main Method
async def chat(user: str, data: dict, args, recommender: RECOMMENDER, metadata: dict, title_to_id: dict, id2entity: dict, save_dir: str, eval_dir: str, entity_list:list, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        global TOTAL_COST
        # Initialize persona
        user_id = user.split('/')[2]
        print(f"Processing User ID: {user_id}")
        persona = data.get("persona", "")
        target_movies = data.get("target movies", [])
        seen_movies = data.get("seen movies", [])
        seeker = UserSimulator(
            args.utterance_prompt_path,
            args.reflection_prompt_path,
            args.api_key,
            args.user_model,
            args.user_temperature,
            args.max_tokens
        )
        # Initialize user info
        await seeker.set_user_info(data)
        
        # For evaluation: save seen+target item list in eval folder
        for_eval = [{
            "seen": seen_movies,
            "target": target_movies
        }]
        
        # For save: record interaction
        dialogue_history = []   # for user
        dialogue_history_unicrs = {  # for crs
            'context': [],
            'resp': "",    
            'rec': [],      
            'entity': []    
        }
        dialogue_for_save = []  # for save
        
        ################################### Init #####################################
        #### 나중에 popularity를 기준으로 뽑는 코드 작성
        recommender_init = "Hello, how can I help you?"
        dialogue_history.append({
            'role': "assistant",
            'content': recommender_init
        })
        dialogue_history_unicrs["context"].append(recommender_init)
        
        ### Grouping seen, unseen, target### 
        seen = []
        unseen = []
        target = []
        candidate_list = []
        
        # turn 0
        save_data = [{
            "persona": persona,
            "seen movies": seen_movies,
            "target movies": target_movies,
        }, {
            "turn 0": recommender_init,
        }]
        #######################################################################################
        
        reflection = ""
        for i in range(args.turn_num):
            turn = i + 1
            
            ###################################### 1. Generate Short-term Memory  ######################################
            ### Generate reflection ###
            f_cost = 0
            if i != 0:
                reflection, f_cost = await seeker.generate_reflection(seen, unseen, target,dialogue_history)
            TOTAL_COST+=f_cost
            
            ###################################### 2. User Interaction ######################################
            ### Generate User Utterance ###
            seeker_response, r_cost = await seeker.generate_utterance(dialogue_history, reflection)
            TOTAL_COST+=r_cost

            dialogue_history_unicrs["context"].append(seeker_response)
            entities = get_entity(seeker_response, entity_list)
            # print(entities)
            dialogue_history_unicrs['entity'] += entities
            dialogue_history_unicrs['entity'] = list(set(dialogue_history_unicrs['entity']))
            # For Model Input
            dialogue_history.append({
                "role": "user",
                'content': seeker_response,
            })
            
            
            ######################################  3. Update Item list ######################################
            # Retrieve Top-50 Recommendations, get_rec returns item ID
            all_recommendations = await recommender.get_rec(dialogue_history_unicrs)
                
            # Convert the ID to String using id2entity
            all_recommendations_str = [f"{id2entity.get(rec_item, 'Unknown')}" for rec_item in all_recommendations]
                
            # Save retrieved top 50 movies for evaluation
            for_eval.append({
                "turn num": turn,
                "recommended": all_recommendations_str
            })  
        
            # Get 4 items for displaying in the item interface
            candidate_list = all_recommendations_str[:4]
        
            # Augment plot of each recommended movie
            candidate_list_with_description = {
                rec_item: {
                    "title": rec_item,
                    "plot": metadata.get(str(title_to_id.get(rec_item, '')), {}).get("plot", "No plot information.")
                }
                for rec_item in candidate_list
            }

            ### Grouping seen, unseen, target and augment plot, reviews of 'seens' ### 
            seen.clear()
            unseen.clear()
            target.clear()
            for item in candidate_list:
                if item in target_movies:
                    target.append(candidate_list_with_description[item])
                elif item in seen_movies:
                    # Find the review for this seen movie
                    for movie in data.get("seen", []):  # Accessing review data
                        if movie.get("title") == item:
                            # Add the movie to the seen list with "title", "plot", and "Your Review"
                            seen.append({
                                "title": item,
                                "plot": metadata.get(str(title_to_id.get(item, '')), {}).get("plot", "No plot information."),
                                "Your Review": movie.get("abstract", "No review provided.")
                            })
                            break  # Exit once the review is found
                else:
                    unseen.append(candidate_list_with_description[item])

            
            ###################################### 2. Recommender Interaction ######################################
            # Assuming recommender.crs_model.get_conv is an async method
            recommender_response = await recommender.crs_model.get_conv(dialogue_history_unicrs)
            # print(recommender_response)
            
            if args.crs_model == 'unicrs':
                recommender_response = clean_unicrs_response(
                    recommender_response, candidate_list, args.dataset
                )
            dialogue_history_unicrs["context"].append(recommender_response)
            entities = get_entity(recommender_response, entity_list)
            dialogue_history_unicrs['entity'] += entities
            dialogue_history_unicrs['entity'] = list(set(dialogue_history_unicrs['entity']))
            # For Model Input
            dialogue_history.append({
                'role': 'assistant',
                # 'content': "Recommender: " + recommender_response,
                'content': recommender_response,
            })
            
            
            ###################################### 4. Record ######################################
            turn_data = {
                f'turn {turn}': {
                    'Reflected_Preferences': reflection,
                    'Seeker': seeker_response,
                    'Recommender': recommender_response,
                    'top_k': candidate_list,
                    'match': {
                        'target': copy.deepcopy(target),
                        'seen': copy.deepcopy(seen),
                        'unseen': copy.deepcopy(unseen)
                    }
                }
            }
            
            # Append the turn data to dialogue_for_save
            dialogue_for_save.append(turn_data)
        
        # Add "dialogue_for_save" at below
        save_data.extend(dialogue_for_save)
        
        ###################################### save data ######################################
        # Save Dialogue
        await save_json_async(f'{save_dir}/{user_id}.json', save_data)
        
        # Save items for evaluation
        await save_json_async(f'{eval_dir}/{user_id}.json', for_eval)

async def main():
    """Main asynchronous function."""
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    global SETTING
    warnings.filterwarnings("ignore")
    
    args = parse_args_unicrs()
    # set_openai_api()
    
    # Directory for saving dialogues and recommendation list.
    save_dir = os.path.join('experiments/5_2_main/results/dialogue_{}_{}_{}'.format(args.turn_num, args.core_num, SETTING, RATIO), args.crs_model, args.kg_dataset)
    eval_dir = os.path.join(save_dir, 'eval')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    random.seed(args.seed)
    
    # Recommender
    model_args = get_model_args(args.crs_model, args)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)
    
    # Load entity and metadata mappings asynchronously
    entity2id_path = os.path.join('crs_data', args.kg_dataset, 'entity2id.json')
    id2info_path = os.path.join('crs_data', args.kg_dataset, 'id2info.json')
    
    entity2id, metadata = await asyncio.gather(
        load_json_async(entity2id_path),
        load_json_async(id2info_path)
    )
    
    # Create mappings
    title_to_id = {
        info['name']: id for id, info in metadata.items() if info['name'] in entity2id
    }         
    id2entity = {int(v): k for k, v in entity2id.items()}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())
    # entity_list = list(entity2id.keys())
    
    # Load user data
    user_data_path = args.user_data_path
    user_data = await load_json_async(user_data_path)
    
    # Random sample for testing
    samples = args.num_samples  # adjust here
    random_selection = random.sample(list(user_data.items()), samples)  # Modify the number for testing
    
    user_data_all = dict(random_selection)
    
    user_data = {}
    # Users, already generated.
    files_in_directory = os.listdir(save_dir)
    files_in_directory = [file[:-5] if file.endswith('.json') else file for file in files_in_directory]
    
    for user_id in user_data_all:
        user_id_short = user_id.split('/')[2]
        if user_id_short not in files_in_directory:
            user_data[user_id] = user_data_all[user_id]
        
    print(f'{len(user_data)} Users left to be generated')
    
     # semaphore
    max_concurrent_users = 50
    semaphore = asyncio.Semaphore(max_concurrent_users)
    # Process each user asynchronously
    tasks = []
    for user, data in user_data.items():
        tasks.append(chat(user, data, args, recommender, metadata, title_to_id, id2entity, save_dir, eval_dir, entity_list, semaphore))
    
    # Await all tasks
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    set_openai_api()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    finally:
        print(f"Total API Cost: ${TOTAL_COST:.6f} USD")