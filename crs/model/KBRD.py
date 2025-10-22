import json
import torch
from collections import defaultdict

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, BartConfig
from transformers import BartConfig

import sys
sys.path.append("..")

from crs.model.kbrd.kg_kbrd import KGForKBRD
from crs.model.kbrd.kbrd_model import KBRDforRec, KBRDforConv
from crs.model.utils import padded_tensor

##### modified, added part#####
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
##### modified, added part#####

class KBRD():
    
    def __init__(self, seed, kg_dataset, 
                 debug, hidden_size, entity_hidden_size, num_bases, 
                 rec_model, conv_model, context_max_length, 
                 tokenizer_path, encoder_layers, decoder_layers, text_hidden_size, attn_head, 
                 resp_max_length, entity_max_length,
                ):
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.kg_dataset = kg_dataset
        # model detailed
        self.debug = debug
        self.hidden_size = hidden_size
        self.entity_hidden_size = entity_hidden_size
        self.num_bases = num_bases
        self.context_max_length = context_max_length
        self.entity_max_length = entity_max_length
        # model
        self.rec_model = f"crs/{rec_model}"
        self.conv_model = f"crs/{conv_model}"
        # conv
        self.tokenizer_path = f"{tokenizer_path}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.text_hidden_size = text_hidden_size
        self.attn_head = attn_head
        self.resp_max_length = resp_max_length
        self.padding = 'max_length'
        self.pad_to_multiple_of = 8
            
        self.kg_dataset_path = f"crs_data/{self.kg_dataset}"
        with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
            self.entity2id = json.load(f)
        
        # Initialize the accelerator.
        self.accelerator = Accelerator(device_placement=False)
        self.device = self.accelerator.device
        
        self.kg = KGForKBRD(kg_dataset=self.kg_dataset, debug=self.debug).get_kg_info()
        self.pad_id = self.kg['pad_id']
        
        # rec model
        self.crs_rec_model = KBRDforRec(
            hidden_size=self.hidden_size,
            num_relations=self.kg['num_relations'], num_bases=self.num_bases, num_entities=self.kg['num_entities'],
        )
        if self.rec_model is not None:
            self.crs_rec_model.load(self.rec_model)
        self.crs_rec_model = self.crs_rec_model.to(self.device)
        self.crs_rec_model = self.accelerator.prepare(self.crs_rec_model)
        
        # conv model
        config = BartConfig.from_pretrained(
            self.conv_model, encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers,
            hidden_size=self.text_hidden_size, encoder_attention_heads=self.attn_head, decoder_attention_heads=self.attn_head,
            encoder_ffn_dim=self.text_hidden_size, decoder_ffn_dim=self.text_hidden_size,
            forced_bos_token_id=None, forced_eos_token_id=None
        )
        
        self.crs_conv_model = KBRDforConv(config, user_hidden_size=self.entity_hidden_size).to(self.device)
        if self.conv_model is not None:
            self.crs_conv_model = KBRDforConv.from_pretrained(self.conv_model, user_hidden_size=self.entity_hidden_size).to(self.device)
        self.crs_conv_model = self.accelerator.prepare(self.crs_conv_model)
    
    async def get_rec(self, conv_dict):
        
        data_dict = {
            'item': [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id],
        }
        
        entity_ids = [self.entity2id[ent] for ent in conv_dict['entity'][-self.entity_max_length:] if ent in self.entity2id],
        
        # kg
        edge_index, edge_type = torch.as_tensor(self.kg['edge_index'], device=self.device), torch.as_tensor(self.kg['edge_type'],
                                                                                                device=self.device)
        
        entity_ids = padded_tensor(
            entity_ids, pad_id=self.pad_id, pad_tail=True, max_length=self.entity_max_length,
            device=self.device, debug=self.debug,
        )
        
        data_dict['entity'] = {
            'entity_ids': entity_ids,
            'entity_mask': torch.ne(entity_ids, self.pad_id)
        }
        
        # infer
        self.crs_rec_model.eval()

        with torch.no_grad():
            data_dict['entity']['edge_index'] = edge_index
            data_dict['entity']['edge_type'] = edge_type
            outputs = self.crs_rec_model(**data_dict['entity'], reduction='mean')

            logits = outputs['logit'][:, self.kg['item_ids']]
            ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
            preds = [[self.kg['item_ids'][rank] for rank in rank_list] for rank_list in ranks]
            labels = data_dict['item']
            
        return preds[0]

    async def get_conv(self, conv_dict, candidate_list):
        
        self.tokenizer.truncation_side = 'left'
        context_list = conv_dict['context']
        context_list.extend(candidate_list)
        context = f'{self.tokenizer.sep_token}'.join(context_list)
        context_batch = self.tokenizer(
            context,
            truncation=True,
            padding=self.padding,
            max_length=self.context_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        ).to(self.device)

        self.tokenizer.truncation_side = 'right'
        resp = conv_dict['resp']
        resp_batch = self.tokenizer(
            resp,
            truncation=True,
            padding=self.padding,
            max_length=self.resp_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        ).to(self.device)
        
        context_batch['labels'] = resp_batch['input_ids']
        
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        
        entity_list = [
            self.entity2id[ent] for ent in conv_dict['entity'][-self.entity_max_length:] if ent in self.entity2id
        ],
        
        entity_ids = padded_tensor(
            entity_list, pad_id=self.pad_id, pad_tail=True, device=self.device,
            debug=self.debug, max_length=self.context_max_length
        )
        
        entity = {
            'entity_ids': entity_ids,
            'entity_mask': torch.ne(entity_ids, self.pad_id),
        }
        
        data_dict = {
            'context': context_batch,
            'entity': entity
        }
        
        edge_index, edge_type = torch.as_tensor(self.kg['edge_index'], device=self.device), torch.as_tensor(self.kg['edge_type'],
                                                                                                device=self.device)
        
        node_embeds = self.crs_rec_model.get_node_embeds(edge_index, edge_type)
        user_embeds = self.crs_rec_model(**data_dict['entity'], node_embeds=node_embeds)['user_embeds']
        
        gen_inputs = {**data_dict['context'], 'decoder_user_embeds': user_embeds}
        gen_inputs.pop('labels')

        gen_args = {
            'min_length': 0,
            'max_length': self.resp_max_length,
            'num_beams': 1,
            'no_repeat_ngram_size': 3,
            'encoder_no_repeat_ngram_size': 3
        }
        gen_seqs = self.accelerator.unwrap_model(self.crs_conv_model).generate(**gen_inputs, **gen_args)
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=True)
        return gen_str