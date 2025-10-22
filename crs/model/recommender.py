import sys
sys.path.append("..")

from crs.model.KBRD import KBRD
from crs.model.BARCOR import BARCOR
from crs.model.UNICRS import UNICRS
from crs.model.CHATGPT import CHATGPT
from crs.model.MACRS import MACRS
from crs.model.ChatCRS import ChatCRS
# from crs.model.MISTRAL import MISTRAL

name2class = {
    'kbrd': KBRD,
    'barcor': BARCOR,
    'unicrs': UNICRS,
    'chatgpt': CHATGPT,
    'macrs': MACRS,
    'chatcrs': ChatCRS,
    # 'mistral': MISTRAL,
}

class RECOMMENDER():
    def __init__(self, crs_model, *args, **kwargs) -> None:
        model_class = name2class[crs_model]
        self.crs_model = model_class(*args, **kwargs)
        
    def get_rec(self, conv_dict):
        return self.crs_model.get_rec(conv_dict)
    
    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)
    
    def init_item_list(self, general_preference): # not used
        return self.crs_model.init_item_list(general_preference)