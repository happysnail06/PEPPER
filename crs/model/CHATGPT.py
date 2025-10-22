import typing
import torch
import json
import os

import nltk
import openai
import tiktoken
import numpy as np

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import RetryCallState, _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm
from openai import AsyncOpenAI

client = AsyncOpenAI()

crs_prompt_path = "prompts/crs_prompt.txt"

def get_prompt(prompt_path : str) -> str:
    if prompt_path:
        with open(prompt_path, 'r') as file:
            prompt = file.read()
    return prompt

async def annotate(conv_str):
    PRICE_PER_1K_INPUT = 0.0001
    
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(conv_str))
    cost = (num_tokens / 1000) * PRICE_PER_1K_INPUT
    
    response = await client.embeddings.create(
        model='text-embedding-ada-002', input=conv_str
    )
    return response.data[0].embedding, cost

async def annotate_chat(messages, logit_bias=None):
    PRICE_PER_1K_INPUT = 0.0005 
    PRICE_PER_1K_OUTPUT = 0.0015
    if logit_bias is None:
        logit_bias = {}
    
    response = await client.chat.completions.create(
        model='gpt-4o-mini', messages=messages, temperature=0, logit_bias=logit_bias,
    )
    
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    
    TOTAL_COST = (prompt_tokens/1000) * PRICE_PER_1K_INPUT + (completion_tokens / 1000) * PRICE_PER_1K_OUTPUT
    
    return response.choices[0].message.content, TOTAL_COST

class CHATGPT():
    
    def __init__(self, seed, debug, kg_dataset) -> None:
        self.seed = seed
        self.debug = debug
        if self.seed is not None:
            set_seed(self.seed)
        
        self.kg_dataset = kg_dataset
        
        self.kg_dataset_path = f"crs_data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
            self.id2info = json.load(f)
            
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        
        self.item_embedding_path = f"crs/model/chatgpt/{self.kg_dataset}"
        
        item_emb_list = []
        id2item_id = []
        files = os.listdir(self.item_embedding_path)
        for file in tqdm(files, desc="Loading item embeddings"):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)

        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
            
        self.chat_recommender_instruction = get_prompt(crs_prompt_path)
    
    async def get_rec(self, conv_dict):
        
        rec_labels = []
        
        context = conv_dict
        context_list = [] # for model
        
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'assistant'
            else:
                role_str = 'user'
            context_list.append({
                'role': role_str,
                'content': text
            })
        
        conv_str = ""
        
        for context in context_list[-2:]:
            conv_str += f"{context['role']}: {context['content']} "
            
        conv_embed, cost = await annotate(conv_str)
        conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        
        
        return item_rank_arr[0], rec_labels, cost
    
    async def get_conv(self, conv_dict, candidate_list:list):
        
        context_list = [] # for model
        context_list.append({
            'role': 'system',
            'content': self.chat_recommender_instruction.format(candidate_list=candidate_list)
        })
        context_list.extend(conv_dict)
        
        gen_str = await annotate_chat(context_list)
        
        return gen_str