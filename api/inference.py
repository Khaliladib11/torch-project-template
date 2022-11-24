import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(BASE_DIR)

from model import Model
import torch
from utils import predict



weights = ''  # path to weights

try:
    model = Model.load_from_checkpoint(weights)

except Exception as e:
    print("There is problem with loading models")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def inference(input):
    return predict(input, model, device)