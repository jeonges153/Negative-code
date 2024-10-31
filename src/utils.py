import os 
import torch
import random
import numpy as np

from sklearn.metrics import f1_score

def save_checkpoint(args, step, model):
    save_path = os.path.join(args.dir_checkpoint, f'proposed_{step}.pt')
    torch.save(model.state_dict(), save_path)
    
    
def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    '''
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    '''

def check_requires_grad(model):
    for name, param in model.named_parameters():
        # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        param.requires_grad = True # ★★★ 
        if param.grad is None:
            print(f"No gradient computed for layer: {name}")
        else:
            print(f"Gradient available for layer: {name}")
        
def metric_accuracy(all_label, all_pred):
    return float(torch.sum(
        torch.Tensor(all_pred) == torch.Tensor(all_label)) 
        / torch.Tensor(all_pred).size()[0]
    )

def metric_f1(all_label, all_pred):
    macro_score = f1_score(all_label, all_pred, average='macro')
    micro_score = f1_score(all_label, all_pred, average='micro')
    weighted_score = f1_score(all_label, all_pred, average='weighted')
    
    return {
        'macro_score': macro_score,
        'micro_score': micro_score,
        'weighted_score': weighted_score
    }