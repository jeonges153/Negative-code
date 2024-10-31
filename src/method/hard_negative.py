import torch
import pandas
import numpy as np 
import ast
import json
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cosine


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
        else:
            return super(NpEncoder, self).default(obj)
       
    
def make_batch(batch):
    batch = batch

    data = pandas.read_csv('./only_3_345711.csv')
    
    origin_category = [3, 4, 5, 7, 11]
    # {2: '장해', 3:'암진단 ', 4: '뇌혈관진단', 5: '심장혈관진단', 6: '질환진단', 7: '치매 간병진단', 8: '골절진단', 9,'치아', 10: '수술 치료비', 11: '입원 통원'}
    
    cate_idx = {3:0, 4:1, 5:2, 7:3, 11:4}
    category = {0: '암진단', 1: '뇌혈관진단', 2: '심장혈관진단', 3: '치매 간병진단', 4: '입원 통원'}
    
    data_len = len(data)
    dataset_label = data.loc[:, 'label']
    
    batch_split = {}
    for i in range(data_len): 
        print("..............  {}번째 iteration 입니다 .............. ".format(i))
        batch_split[i] = {}
        
        '''
        각 batch마다 random하게 1개의 pos 추출, batch-1개 만큼 neg 추출
        
        pos와 neg의 euclidean distance를 구해서, 가까울 수록 더 많이 뽑도록 batch를 구성함
        
        '''
        
        # pos1 (query)
        pos1_idx = i # pos1_idx = np.random.randint(0, data_len)
        pos1_embed = data.loc[pos1_idx, 'embedding']
        pos1_embed = torch.tensor(ast.literal_eval(pos1_embed)).reshape(1, -1) # (768) -> (1, 768)
        pos1_label = data.loc[pos1_idx, 'label'].tolist()
        
        # pos2 (key)
        pos2_idx_range = (np.where(dataset_label == pos1_label)[0]).tolist()
        if i in pos2_idx_range: # 자기자신 지우기
            pos2_idx_range.remove(i)
        pos2_embed_range = data.loc[pos2_idx_range, 'embedding'] 
        pos2_embed_range_list = [torch.tensor(ast.literal_eval(n)) for n in pos2_embed_range]
        pos2_embed_range = torch.stack(pos2_embed_range_list) 
            
        pos_distance = euclidean_distances(pos2_embed_range, pos1_embed).flatten()
        pos_distance_sort = sorted(enumerate(pos_distance), key=lambda x: x[1])
        
        pos2_idx, d = list(zip(*pos_distance_sort)) #  가장 가까운 sample을 positive라고 간주함
        pos2_idx = np.array(pos2_idx_range[0]).tolist()
        pos2_embed = data.loc[pos2_idx, 'embedding']
        pos2_embed = torch.tensor(ast.literal_eval(pos2_embed)).reshape(1, -1)
        pos2_label = data.loc[pos2_idx, 'label'].tolist()
        
        # print(pos2_idx)
        # print(pos2_embed.shape)
        # print(pos2_label)
        
        # neg (batch-2만큼)
        neg1_idx_range = (np.where(dataset_label != pos1_label)[0]).tolist() 
        neg1_embed_range = data.loc[neg1_idx_range, 'embedding'] 
        neg1_label_range = data.loc[neg1_idx_range, 'label']
        neg1_embed_range_list = [torch.tensor(ast.literal_eval(n)) for n in neg1_embed_range]
        neg1_embed_range = torch.stack(neg1_embed_range_list) 

        neg_distance = euclidean_distances(neg1_embed_range, pos1_embed).flatten()
        neg_distance_sort = sorted(enumerate(neg_distance), key=lambda x: x[1]) # ** 여기서 enumerate 하니까 문제임 ~~~ 
        
        neg_idx, d = list(zip(*neg_distance_sort))
        neg_idx = np.array(neg_idx[:batch-2]).tolist()
        neg_idx = [neg1_idx_range[i] for i in neg_idx]
        
        neg_embed = data.loc[neg_idx, 'embedding']
        neg_embed = [torch.tensor(ast.literal_eval(n)) for n in neg_embed]
        neg_embed = torch.stack(neg_embed)
        neg_label = data.loc[neg_idx, 'label'].tolist()

        # print(neg_idx)
        # print(neg_embed.shape)
        # print(neg_label)
        
        idx = [pos1_idx] + [pos2_idx] + neg_idx
        embedding = torch.cat([pos1_embed, pos2_embed], dim=0)
        embedding = torch.cat([embedding, neg_embed], dim=0)
        label = [pos1_label] + [pos2_label] + neg_label
        print(label)
        label = [cate_idx[lb] for lb in label] # 모델 학습용 label로 바꿔주기
        
        # print(idx)
        # print(label)
        # print(embedding.shape)
        
        
        batch_split[i]['idx'] = idx
        batch_split[i]['label'] = label
        batch_split[i]['embedding'] = embedding.numpy()
        ############################################################################################################

    with open("./hard_negative_345711.json", 'w') as json_file:
        json.dump(batch_split, json_file, indent=4, cls=NpEncoder)

make_batch(8)