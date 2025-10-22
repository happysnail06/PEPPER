import json
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from collections import defaultdict

import logging
                # self.crs_rec_model = BartForSequenceClassification.from_pretrained(self.rec_model, num_labels=self.kg['num_entities']).to(self.device)
import sys
sys.path.append("..")

from crs.model.barcor.kg_bart import KGForBART
from crs.model.barcor.barcor_model import BartForSequenceClassification

class BARCOR():
    
    def __init__(self, seed, kg_dataset, debug, tokenizer_path, context_max_length,
                 rec_model, conv_model,
                 resp_max_length):
        logging.getLogger('transformers').setLevel(logging.ERROR)
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.kg_dataset = kg_dataset
        
        self.debug = debug
        self.tokenizer_path = f"{tokenizer_path}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        
        self.padding = 'max_length' 
        self.pad_to_multiple_of = 8 
        
        self.accelerator = Accelerator(device_placement=False)
        self.device = self.accelerator.device
        
        self.rec_model = f"crs/{rec_model}"
        self.conv_model = f"crs/{conv_model}"
        
        # conv
        self.resp_max_length = resp_max_length
        
        self.kg = KGForBART(kg_dataset=self.kg_dataset, debug=self.debug).get_kg_info()
        
        self.crs_rec_model = BartForSequenceClassification.from_pretrained(self.rec_model, num_labels=self.kg['num_entities']).to(self.device)
        self.crs_conv_model = AutoModelForSeq2SeqLM.from_pretrained(self.conv_model).to(self.device)
        self.crs_conv_model = self.accelerator.prepare(self.crs_conv_model)
        
        self.kg_dataset_path = f"crs_data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        
        
        
    async def get_rec(self, conv_dict):
        pass
        
        # dataset
        text_list = []
        turn_idx = 0
        
        for utt in conv_dict:  # modifed, conv_dict['context'] -> conv_dict
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'System: '
                else:
                    text += 'User: '
                text += utt
                text_list.append(text)
            turn_idx += 1
        
        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

        data_list = []
        
        data_dict = {
            'context': context_ids,
        }
        data_list.append(data_dict)
        
        # dataloader
        input_dict = defaultdict(list)
        
        for data in data_list:
            input_dict['input_ids'].append(data['context'])
        
        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        
        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device)
                
        self.crs_rec_model.eval()
        outputs = self.crs_rec_model(**input_dict) 
        item_ids = torch.as_tensor(self.kg['item_ids'], device=self.device)
        logits = outputs['logits'][:, item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids[ranks].tolist()
        
        return preds[0]
    
    async def get_conv(self, conv_dict, candidate_list):
        
        text_list = []
        turn_idx = 0
        for utt in conv_dict:
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'System: '
                else:
                    text += 'User: '
                text += utt
                text_list.append(text)
            turn_idx += 1
        # text_list.extend(candidate_list)
        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)


        data_dict = {
            'context': context_ids,
        }
        
        input_dict = defaultdict(list)
        
        input_dict['input_ids'] = data_dict['context']
        
        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        
        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
                
        self.crs_conv_model.eval()
        
        gen_args = {
            'min_length': 0,
            'max_length': self.resp_max_length,
            'num_beams': 1,
            'no_repeat_ngram_size': 3,
            'encoder_no_repeat_ngram_size': 3
        }
        
        gen_seqs = self.accelerator.unwrap_model(self.crs_conv_model).generate(**input_dict, **gen_args)
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=True)
        
        return gen_str