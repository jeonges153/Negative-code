import torch
import json
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from model import ContrastiveModel, ClassificationModel


class One_TrainDataset(Dataset):
    def __init__(self, args):
        super(One_TrainDataset, self).__init__()
        with open(args.dir_data, "r", encoding="utf-8") as f:
            self.data_split = json.load(f)
                        
    def __len__(self):
        return len(self.data_split)
    
    def __getitem__(self, idx):
        index = self.data_split[str(idx)]['idx']
        text = self.data_split[str(idx)]['text']
        label = self.data_split[str(idx)]['label']
        return index, text, label
    
    
class One_EvalDataset(Dataset): 
    def __init__(self, args):
        super(One_EvalDataset, self).__init__()
        with open(args.dir_data, "r", encoding="utf-8") as f:
            self.data_split = json.load(f)
      
    def __len__(self):
        return len(self.data_split)
    
    def __getitem__(self, idx):
        index = self.data_split[str(idx)]['idx']
        text = self.data_split[str(idx)]['text']
        label = self.data_split[str(idx)]['label']
        return index, text, label


class One_TestDataset(Dataset): 
    def __init__(self, args):
        super(One_TestDataset, self).__init__()
        
        self.df = pd.read_csv(args.inference_data, encoding='UTF8')
        
        # {2: '장해', 3:'암진단 ', 4: '뇌혈관진단', 5: '심장혈관진단', 6: '질환진단', 7: '치매 간병진단', 8: '골절진단', 10: '수술 치료비', 11: '입원 통원'}
        label = args.category_list
        self.cate2idx = {c:idx for idx, c in enumerate(label)}
        after_category = {0: '암진단',  1: '뇌혈관진단', 2: '심장혈관진단', 3: '치매 간병진단', 4: '입원 통원'}

        self.model = ContrastiveModel()
        # self.model.load_state_dict(torch.load('./checkpoint/hard_negative_layer2_0.pt'))
        
                
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        text = self.df.loc[idx, 'query']
        label = int(self.df.loc[idx, 'label'])
        label = self.cate2idx[label]
        
        embedding = self.model.model.encode(text) # nn.Embedding layer 통과하여 얻은 임베딩
        
        return len(self.df), text, embedding, label
    

def build_loader(args, dataset): # batch가 1만큼 들어가야겠네 ··· 
    dataloader = DataLoader(dataset, batch_size=int(args.batch_size/args.batch_size), shuffle=args.shuffle)
    return dataloader
    