#%%

import torch
import pandas
from torch import nn
from torch import einsum
import torch.nn.functional as F

# from embedding import EmbeddingModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class ContrastiveModel(nn.Module):
    def __init__(self, num_class=5):
        super(ContrastiveModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
        self.model_ =  AutoModel.from_pretrained("jhgan/ko-sroberta-multitask")
        
        '''
        print("\n--- Transformer Module Details ---")
        transformer = self.model_[0].auto_model
        for name, param in transformer.named_parameters():
            print(f"Layer: {name}, Shape: {param.shape}")

        # Pooling 모듈의 세부 설정 확인
        print("\n--- Pooling Module Details ---")
        pooling = self.model_[1]
        print(f"Pooling Mode CLS Token: {pooling.pooling_mode_cls_token}")
        print(f"Pooling Mode Mean Tokens: {pooling.pooling_mode_mean_tokens}")
        print(f"Pooling Mode Max Tokens: {pooling.pooling_mode_max_tokens}")
        print(f"Word Embedding Dimension: {pooling.word_embedding_dimension}")
        '''
        # for p in self.model.parameters(): #### backpropagation을 위함 ####
        #     p.requires_grad=True
        self.dim = self.model_.embeddings.word_embeddings.embedding_dim
        # self.linearlayer = nn.Linear(self.dim, self.dim)
        # self.activation = nn.ReLU() 
        self.classifier = nn.Linear(self.dim, num_class)
        
        self.tau = 0.1
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
    
    def forward(self, idx, text, label):
        text = [t[0] if isinstance(t, tuple) else t for t in text]
        tok = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        embeds = self.model_(
            tok['input_ids'],
            attention_mask=tok['attention_mask']
        )
        embedding = embeds.pooler_output # torch.Size([8, 768]) 
        label = torch.tensor(label) # torch.Size([8])
        
        
        # contrastive learning
        # embedding = F.normalize(embedding, p=2, dim=1)
        pos1 = embedding[0].unsqueeze(0) # 1, 768
        pos2 = embedding[1].unsqueeze(0) # 1, 768
        neg = embedding[2:] # 6, 1024
        
        pos_score = self.similarity_fct(pos1, pos2) / self.tau 
        neg_score = self.similarity_fct(pos1, neg) / self.tau 
                
        max_neg_score = torch.max(neg_score, dim=0)[0] # 가장 큰 neg score
        max_score = torch.max(pos_score, max_neg_score) # score 중에 가장 큰 
        
        if len(max_score.shape) == 1:
            max_score = max_score.unsqueeze(1)
        elif len(max_score.shape) == 0:
            max_score = max_score.unsqueeze(0)
            
        stable_pos_scores = pos_score - max_score # 0 
        stable_neg_scores = neg_score - max_score.unsqueeze(1) # 음수 
        
        # 확률값 얻기 위함 -> # 가까울수록 더 높은 확률 값을 얻게 됨 
        exp_pos_scores = torch.exp(stable_pos_scores) 
        exp_neg_scores = torch.exp(stable_neg_scores)

        total_scores_sum = exp_pos_scores + exp_neg_scores.sum(dim=1)
        
        log_prob = torch.log(exp_pos_scores / total_scores_sum)
        contrastive_loss = -log_prob.mean()
        
        # out = self.linearlayer(embedding)
        # out = self.activation(out)
        classification_out = self.classifier(embedding)
        return classification_out, contrastive_loss, label
    

class ClassificationModel(nn.Module): 
    def __init__(self):
        super(ClassificationModel, self).__init__()
        # self.model = EmbeddingModel().load()
        # print(self.model[0]) # Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: XLMRobertaModel
        # print(self.model[0]._modules) # XLMRobertaEmbeddings, XLMRobertaEncoder, XLMRobertaPooler
        dim = self.model_.embeddings.word_embeddings.embedding_dim
        self.classifier = nn.Linear(dim, 5) # 1024, 8
        
        
    def forward(self, idx, label, embedding):
        label = torch.tensor(label)
        embedding = embedding.squeeze(0) 
        # [1, 8, 1024] -> [8, 1024]
        classification_out = self.classifier(embedding)
        return classification_out, label
    

def build_model(args):
    model = ContrastiveModel(num_class=args.num_class)
    
    return model