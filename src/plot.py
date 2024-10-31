#%%
import ast
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model import ContrastiveModel


df = pd.read_csv('./only_3_345711.csv', encoding='UTF8') 

model = ContrastiveModel()
checkpoint = torch.load('./checkpoint/ckpt_47.pt')
model.load_state_dict(checkpoint)

plt.figure(figsize=(8, 6))
plt.title('PCA 2D Projection')

colors = ['blue', 'green', 'orange', 'purple', 'pink']
labels = [3, 4, 5, 7, 11]
np_array_list = []
labels_list = []

for i, c in enumerate(labels):
    for idx, row in df[df['label']==c].iterrows():
        print(f"Processing index: {idx}")
        embeds = torch.tensor(ast.literal_eval(row['embedding']))
        classification_out, contrastive_loss, label = model(None, row['label'], embeds)
        
        # embedding = classification_out # 임베딩을 NumPy 배열로 변환
        np_array_list.append(classification_out.detach().numpy())
        labels_list.append(c)
        
np_array_list = np.array(np_array_list) # NumPy 배열로 변환
    # centroid = np.mean(np_array_list, axis=0).reshape(1, -1) # centroid 구하기
    # embeddings_with_centroid = np.vstack([np_array_list, centroid]) # centroid랑 embedding
    # centroid_pca = pca.transform(embeddings_with_centroid)
pca = PCA(n_components=2)
pca.fit(np_array_list)
embeddings_pca = pca.transform(np_array_list)

for i, c in enumerate(labels): # scatter plot을 class 별로 시각화 
    plt.scatter(embeddings_pca[np.array(labels_list) == c, 0], 
                embeddings_pca[np.array(labels_list) == c, 1], 
                c=colors[i], label=f'Class {c} Embeddings')
plt.legend()
plt.show()
