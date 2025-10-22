import json
import os
import argparse


def set_openai_api():
    personal_info = json.load(open("config/personal_info.json", "r"))
    os.environ["OPENAI_API_KEY"] = personal_info["api_key"]
    os.environ["OPENAI_ORGANIZATION"] = personal_info["org_id"]
    print(f"Set OpenAI API Key.")
    
def get_prompt(prompt_path : str) -> str:
    if prompt_path:
        with open(prompt_path, 'r') as file:
            prompt = file.read()
    return prompt
    
def parse_args_chatgpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)

    #### For Recommender ####
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'chatcrs'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    
    args = parser.parse_args()
    return args

def parse_args_chatcrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)

    #### For Recommender ####
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'chatcrs'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    
    args = parser.parse_args()
    return args

def parse_args_macrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)

    #### For Recommender ####
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'macrs', 'chatcrs'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    
    args = parser.parse_args()
    return args

def parse_args_barcor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)
    
    #### For Recommender ####
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'chatcrs'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--resp_max_length', type=int)
    parser.add_argument('--tokenizer_path', type=str)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    args = parser.parse_args()
    return args

def parse_args_unicrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)
    
    #### For Recommender ####
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'chatcrs'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    
    parser.add_argument('--model', type=str)
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)
    parser.add_argument('--resp_max_length', type=int)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--text_tokenizer_path', type=str)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--num_bases', type=int, default=8)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    args = parser.parse_args()
    return args

def parse_args_kbrd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    #### For User Simulator ####
    parser.add_argument("--user_data_path", type=str, required=True, help="The directory to the user data.")
    parser.add_argument("--user_model", type=str, default="gpt-3.5-turbo", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--user_temperature", type=float, default=1.0)
    
    parser.add_argument("--utterance_prompt_path", type=str, default=None)
    parser.add_argument("--reflection_prompt_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--core_num', type=int, default=5)
    
    #### For Recommender ####
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt', 'chatcrs'])
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--entity_hidden_size', type=int)
    parser.add_argument('--num_bases', type=int, default=8)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--encoder_layers', type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--attn_head', type=int)
    parser.add_argument('--text_hidden_size', type=int)
    parser.add_argument('--resp_max_length', type=int)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--num_samples', type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    args = parser.parse_args()
    return args


def get_model_args(model_name, args):
    if model_name == 'kbrd':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'hidden_size': args.hidden_size,
            'entity_hidden_size': args.entity_hidden_size, 'num_bases': args.num_bases,
            'rec_model': args.rec_model, 'conv_model': args.conv_model,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length, 'tokenizer_path': args.tokenizer_path,
            'encoder_layers': args.encoder_layers, 'decoder_layers': args.decoder_layers, 'text_hidden_size': args.text_hidden_size,
            'attn_head': args.attn_head, 'resp_max_length': args.resp_max_length,
            'seed':args.seed
        }
    elif model_name == 'barcor':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'rec_model': args.rec_model, 'conv_model': args.conv_model, 'context_max_length': args.context_max_length,
            'resp_max_length': args.resp_max_length, 'tokenizer_path': args.tokenizer_path, 'seed': args.seed
        }
    elif model_name == 'unicrs':
        args_dict = {
            'debug': args.debug, 'seed': args.seed, 'kg_dataset': args.kg_dataset, 'tokenizer_path': args.tokenizer_path,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length, 'resp_max_length': args.resp_max_length,
            'text_tokenizer_path': args.text_tokenizer_path,
            'rec_model': args.rec_model, 'conv_model': args.conv_model, 'model': args.model, 'num_bases': args.num_bases, 'text_encoder': args.text_encoder
        }
    elif model_name == 'chatgpt' or model_name == 'macrs' or model_name == 'chatcrs':
        args_dict = {
            'seed': args.seed, 'debug': args.debug, 'kg_dataset': args.kg_dataset
        }
    else:
        raise Exception('do not support this model')
    
    return args_dict