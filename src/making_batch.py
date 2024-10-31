import ast
import json
import torch
import pandas as pd
import numpy as np 

from sklearn.metrics.pairwise import euclidean_distances

'''
one_embedding, two_embedding 구성을 위한 코드

'''

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, pd.Series):
        #     return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
       
    
def make_batch(args):
    batch = args.batch_size

    data = pd.read_csv(args.from_csv)
    
    # {2: '장해', 3:'암진단 ', 4: '뇌혈관진단', 5: '심장혈관진단', 6: '질환진단', 7: '치매 간병진단', 8: '골절진단', 9,'치아', 10: '수술 치료비', 11: '입원 통원'}
    origin_category = args.category_list
    cate2idx = {c:idx for idx, c in enumerate(origin_category)}
    # category = {0: '암진단', 1: '뇌혈관진단', 2: '심장혈관진단', 3: '치매 간병진단', 4: '입원 통원'}
    
    data_len = len(data)
    dataset_label = data.loc[:, 'label']

    # for i in range(data_len):
    #     for j in range(data_len):
    #         if i == j:
    #             continue
    #         emb1 = torch.tensor(ast.literal_eval(data.loc[i, 'embedding']))
    #         emb2 = torch.tensor(ast.literal_eval(data.loc[j, 'embedding']))
    #         angle = np.degrees(np.arccos(1- cosine(emb1, emb2)))
    #         if angle <= 5:
    #             print(angle)
    #             # data = data.drop(j, axis=0)
    
    # print(len(data))
    
    batch_split = {}
    for i in range(data_len): 
        print(i)
        batch_split[i] = {}
        
        ''' 각 batch마다 random하게 1개의 pos 추출, batch-1개 만큼 neg 추출
        pos와 neg의 euclidean distance를 구해서, 가까울 수록 더 많이 뽑도록 batch를 구성함 '''
        
        # pos1 (query)
        pos1_idx = i # pos1_idx = np.random.randint(0, data_len)
        pos1_text = data.loc[pos1_idx, 'content']
        pos1_embed = data.loc[pos1_idx, 'embedding']
        pos1_embed = torch.tensor(ast.literal_eval(pos1_embed)).reshape(1, -1) # (768) -> (1, 768)
        pos1_label = data.loc[pos1_idx, 'label'].tolist()
        
        # pos2 (key)
        pos2_idx_range = (np.where(dataset_label == pos1_label)[0]).tolist()
        pos2_embed_range = data.loc[pos2_idx_range, 'embedding'] 
        pos2_embed_range_list = [torch.tensor(ast.literal_eval(n)) for n in pos2_embed_range]
        pos2_embed_range = torch.stack(pos2_embed_range_list) 
            
        pos_distance = euclidean_distances(pos2_embed_range, pos1_embed).flatten()
        pos_probabilties = np.array([1/d if d != 0 else 0.0 for d in pos_distance])
        pos_probabilties /= pos_probabilties.sum()
        
        pos2_idx = int(np.random.choice(pos2_idx_range, size=1, p=pos_probabilties)[0]) # pos idx 아무거나 뽑기 
        pos2_text = data.loc[pos2_idx, 'content']
        # pos2_embed = data.loc[pos2_idx, 'embedding']
        # pos2_embed = torch.tensor(ast.literal_eval(pos2_embed)).reshape(1, -1)
        pos2_label = data.loc[pos2_idx, 'label'].tolist()
        
        # neg1 (batch-1만큼)
        neg1_idx_range = (np.where(dataset_label != pos1_label)[0]).tolist() 
        neg1_embed_range = data.loc[neg1_idx_range, 'embedding'] 
        neg1_label_range = data.loc[neg1_idx_range, 'label']
        neg1_embed_range_list = [torch.tensor(ast.literal_eval(n)) for n in neg1_embed_range]
        neg1_embed_range = torch.stack(neg1_embed_range_list) 

        # distance에 따라 추출하는 확률 다르게 계산하기 위함 (가까울수록 더 많이 뽑기)
        neg_distance = euclidean_distances(neg1_embed_range, pos1_embed).flatten()
        neg_probabilties = np.array([1/d if d != 0 else 0.0 for d in neg_distance])
        neg_probabilties /= neg_probabilties.sum()
        
        # 다른 label을 갖는 idx 중에 'batch-2'개를 neg_probabilities 확률로 무작위 추출
        neg1_idx = list(np.random.choice(neg1_idx_range, size=batch-2, p=neg_probabilties)) # neg idx 아무거나 뽑기 
        neg1_text = data.loc[neg1_idx, 'content']
        neg1_text = [t for t in neg1_text]
        # neg1_embed = data.loc[neg1_idx, 'embedding']
        # neg1_embed = [torch.tensor(ast.literal_eval(n)) for n in neg1_embed]
        # neg1_embed = torch.stack(neg1_embed)
        neg1_label = data.loc[neg1_idx, 'label'].tolist()
        
        idx = [pos1_idx] + [pos2_idx] + neg1_idx
        text = [pos1_text] + [pos2_text] + neg1_text
        # embedding = torch.cat([pos1_embed, pos2_embed], dim=0)
        # embedding = torch.cat([embedding, neg1_embed], dim=0)
        label = [pos1_label] + [pos2_label] + neg1_label
        label = [cate2idx[lb] for lb in label] # 모델 학습용 label로 바꿔주기
        
        # print(idx)
        # print(label)
        # print(embedding.shape)
        
        batch_split[i]['idx'] = idx
        batch_split[i]['text'] = text
        batch_split[i]['label'] = label
        ############################################################################################################
    
    with open(args.dir_data, 'w', encoding="utf-8") as json_file:
        json.dump(batch_split, json_file, ensure_ascii=False, indent=4, cls=NpEncoder)
        