import configparser, os, logging
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class EmbeddingModel(nn.Module):
    def __init__(self, model = 'sentenceTransformer'):
        self._name = "jhgan/ko-sroberta-multitask"

    def load(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
        if not torch.cuda.is_available():
            logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
            model = SentenceTransformer(self._name)
        else:
            model = SentenceTransformer(self._name, device="cuda")
        if not os.path.exists(self._name):
            model.save(self._name)
        return model