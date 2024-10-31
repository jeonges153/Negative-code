import os
import json
import wandb
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from args import parse_args
from model import build_model
from making_batch import make_batch
from dataset import build_loader, One_TrainDataset, One_EvalDataset 
from utils import (
    set_seed, 
    check_requires_grad, 
    save_checkpoint, 
    metric_f1, 
    metric_accuracy
)


from datetime import datetime


wandb.init(
    project="contrastive learning", 
    name="proposed_layer2"
)

# def save_model(model, path): 
#     torch.save(model.state_dict(), path)
#     print(f"Model saved to {path}")
    
def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.dir_data: 
        make_batch(args) # batch size만큼 pos, neg 만들기 ★★★
    
    os.makedirs(args.dir_checkpoint, exist_ok=True) # './results/save_model'

    train_dataloader = build_loader(args, One_TrainDataset(args))
    eval_dataloader = build_loader(args, One_EvalDataset(args))
    
    model = build_model(args)
    check_requires_grad(model)
    
    for p in model.parameters(): #### backpropagation을 위함 ####
        p.requires_grad=True
    
    optimizer = Adam(
        params=model.parameters(), 
        lr=args.lr_base
    )
    criterion = nn.CrossEntropyLoss()
        
    d = {}
    best_accuracy = 0
    for E in range(args.epoch):
        d[E] = {}
        model.train()
        
        train_loss_list = []
        all_train_pred = []
        all_train_label = []
        
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            idx = data[0]
            text = data[1]
            label = data[2] 
            
            classification_out, contrastive_loss, label = model(idx, text, label)
                    
            classification_loss = criterion(classification_out, label)
            pred = torch.argmax(classification_out, dim=1)
            
            loss = classification_loss + contrastive_loss * args.alpha
            
            all_train_pred.extend(pred)
            all_train_label.extend(label)
            
            loss.backward() # 최종 loss back propagation
            optimizer.step()
            
            train_loss_list.append(loss.item())

        train_loss = sum(train_loss_list) / len(train_loss_list)
        print("train loss in epoch {}: {:.2f}" .format(str(E), train_loss))
        
        train_acc = metric_accuracy(all_train_label, all_train_pred)
        train_f1 = metric_f1(all_train_label, all_train_pred)
        print("train acc in epoch {}: {:.2f}".format(E, train_acc))
        for name, value in train_f1.items(): 
            print("train {} in epoch {}: {:.2f}" .format(name, E, value))

        wandb.log({
            'train_loss': train_loss, 
            'train_acc': train_acc, 
            'train_macro_f1': train_f1['macro_score'],
            'train_micro_f1': train_f1['micro_score'],
            'train_weight_f1': train_f1['weighted_score']
        })
        
        d[E]['train_loss'] = round(train_loss, 2) 
        d[E]['train_acc'] = round(train_acc, 2) 
        d[E]['correct'] = str(int(torch.sum(torch.Tensor(all_train_pred) == torch.Tensor(all_train_label)))) + " / " + str(len(all_train_pred))
        d[E]['pred/label'] = list(zip(np.array(all_train_pred).tolist(), np.array(all_train_label).tolist()))

        ##########################################################################################################################################
        test_loss_list = []
        all_test_pred = []
        all_test_label = []
        
        print("This is validation step ··· ")
        check_requires_grad(model)
        quit()
        
        with torch.no_grad(): 
            model.eval()
            for data in tqdm(eval_dataloader):
                idx = data[0] 
                label  = data[1] 
                embedding = data[2] 
                
                classification_out, contrastive_loss, label = model(idx, label, embedding) 
                
                classification_loss = criterion(classification_out, label)
                pred = torch.argmax(classification_out, dim=1)

                loss = classification_loss + contrastive_loss * args.alpha
                
                all_test_pred.extend(pred)
                all_test_label.extend(label)
                
                test_loss_list.append(loss.item())
            
        test_loss = sum(test_loss_list) / len(test_loss_list)
        print("test loss in epoch {}: {:.2f}" .format(str(E), test_loss))
        test_acc = metric_accuracy(all_test_label, all_test_pred) 
        test_f1 = metric_f1(all_test_label, all_test_pred)
        print("test acc in epoch {}: {:.2f}".format(E, test_acc))
        for name, value in test_f1.items(): 
            print("test {} in epoch {}: {:.2f}" .format(name, E, value))
        
        wandb.log({
            'test_loss': test_loss, 
            'test_acc': test_acc, 
            'test_macro_f1': test_f1['macro_score'],
            'test_micro_f1': test_f1['micro_score'],
            'test_weight_f1': test_f1['weighted_score']
        })
        
        d[E]['test_loss'] = round(test_loss, 2) 
        d[E]['test_acc'] = round(test_acc, 2) 
        d[E]['test_correct'] = str(int(torch.sum(torch.Tensor(all_test_pred) == torch.Tensor(all_test_label)))) + " / " + str(len(all_test_pred))
        d[E]['test_pred/label'] = list(zip(np.array(all_test_pred).tolist(), np.array(all_test_label).tolist()))
        
        
        # best_acc 갱신
        if best_accuracy < round(test_acc, 2):
            best_accuracy = round(test_acc, 2) 
            save_checkpoint(args, E, model)
            
            print("############################ {} epoch best_accuracy ############################\n\n".format(E))

        
        json.dump(d, open(os.path.join('./results', "proposed_layer2_result.json"), "w"), indent=2) # 결과 저장파일
            
if __name__ == '__main__':
    main()