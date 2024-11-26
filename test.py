import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
import json
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch import nn,optim
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from main import test


model = get_net()
model = torch.nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu') 
    
model.to(device)

optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss().to(device) 
test_files = get_files(config.test_data,"test") 
# Checkpoint to load the best model.
best_model = torch.load(config.best_models + os.sep+config.model_name+os.sep+ str(0) +os.sep+ 'model_best.pth.tar')
# Load the state dictionary of the best model.
model.load_state_dict(best_model["state_dict"])
test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
# Call the test function for testing.
test(test_dataloader,model,1)



#  baseline.json
with open('./submit/test_result.json', 'r') as f:
    results = json.load(f)

# Extract categories
classes = [item['disease_class'] for item in results]

# Statistical category distribution
class_counts = Counter(classes)

# Draw a bar chart
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Disease Class")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Disease Classes")
plt.savefig('./class_distribution.png', dpi=300, bbox_inches='tight')
