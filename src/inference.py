import torch
import wandb
import os
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
from torch import nn

from model import ContrastiveModel, ClassificationModel
from dataset import One_TestDataset
from torch.utils.data import DataLoader 


wandb.init(
    project="contrastive learning", 
    name="hard_negative_layer2_inference"
)


inference_batch = 8

test_dataset = One_TestDataset()
inference_dataloader = DataLoader(test_dataset, batch_size=inference_batch, shuffle=True)
criterion = nn.CrossEntropyLoss()

def inference():
    checkpoint = './checkpoint/hard_negative_layer2_0.pt'
    
    model = ContrastiveModel() # 'classifier.weight' / 'classifier.bias'
    # model.load_state_dict(torch.load(checkpoint))
    
    inference_acc_list = []
    label_count = []
    pred_count = []
    inference_pred = []
    inference_label = []
    
    df = pd.DataFrame(columns=['text', 'probability', 'prob', 'label'])
    idx = 0
    
    d = {}
    with torch.no_grad():
        model.eval()
        
        probabiltiy = {}
        for data in tqdm(inference_dataloader):
            length = data[0]
            text = data[1]
            embeddings = data[2] 
            label = data[3] 
            
            print(embeddings,'\n\n') # 초기 임베딩  
            
            print(model(None, label, embeddings),'\n\n') # .forward로 들어가기 떄문에 -> label, embedding 까지 함께 argument로 넣어줘야함
            print(model.testing(None, label, embeddings),'\n\n')
            
            model.load_state_dict(torch.load(checkpoint))
            print(model(None, label, embeddings),'\n\n')
            quit()
            
            classification_out, label = model.testing(None, label, embeddings) 
            classification_loss = criterion(classification_out, label)
            
            prob = nn.functional.softmax(classification_out, dim=1) #(batch, num_class)
            prob = (prob * 100).int()
            
            for i, prob_zip in enumerate(prob):
                probabiltiy[i] = []
                for j, probb in enumerate(prob_zip):
                    probabiltiy[i].append({j: probb.item()})
                # print(probabiltiy[i])


            for t, pp, p, l in list(zip(text, prob, probabiltiy.values(), label)):
                df.loc[idx, 'text'] = t
                df.loc[idx, 'probability'] = pp
                df.loc[idx, 'prob'] = p
                df.loc[idx, 'label'] = l
                print(idx, "\t", pp, "\t",  t,": \t", p, "\t", l)
                idx += 1
            
            # top-3 예측 클래스 얻기
            topk_values, topk_indices = torch.topk(classification_out, k=3, dim=1)
            
            pred = torch.argmax(classification_out, dim=1) # top-1
            # top-3 클래스의 인덱스 중 실제 라벨이 포함되어 있는지 확인
            correct = (topk_indices == label.unsqueeze(1)).any(dim=1)
            print(correct)

            # 정답 개수 계산
            num_correct = correct.sum().item()
            print(num_correct)
            print()

            loss = classification_loss
            
            inference_pred.extend(topk_indices)
            inference_label.extend(label)
            
            inference_acc_list.append(num_correct)
    
    df.to_csv('./345711_inference_results.csv')
    
    print(sum(inference_acc_list))
    inference_accuracy = sum(inference_acc_list) / length[0].item() * 100
    print("inference accuracy in epoch {}: {:.2f}" .format(str(1), inference_accuracy))
    
    wandb.log({
        # 'inference_loss': test_accuracy, 
        'inference_acc': inference_accuracy
    })
    
    d['inference_loss'] = round(loss.item(), 2)
    d['inference_accuracy'] = round(inference_accuracy, 2)
    d['inference_text'] = [t for t in text.tolist()]
    d['inference_pred/label'] = list(zip(np.array(inference_pred).tolist(), np.array(inference_label).tolist()))

    json.dump(d, open(os.path.join('./results', "345711_layer2_inference.json"), "w"), indent=2) # 결과 저장파일


inference()