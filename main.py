import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
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

#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
CUDA_LAUNCH_BLOCKING=2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

device = torch.device("cuda:0")  
print(f"Current device index: {device}")  

#2. evaluate func
def evaluate(val_loader,model,criterion):
    """
    val_loader is a data loader used to provide data for the validation set; model is the CNN model to be evaluated; criterion is the loss function, usually used to calculate the gap between the model output and the true label.
    """
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.cuda()
    model.eval() 
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader): 
            input = Variable(input).cuda() 
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            #2.2.1 compute output
            output = model(input) 
            loss = criterion(output,target) 

            #2.2.2 measure accuracy and record loss
            # precision1, precision2 = accuracy(output, target, topk=(1, 2))
            precision1,precision2 = accuracy(output,target,topk=(1,2))
            losses.update(loss.item(),input.size(0))
            top1.update(precision1[0],input.size(0))
            top2.update(precision2[0],input.size(0)) 

    return [losses.avg,top1.avg,top2.avg]

#3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    #3.1 confirm the model converted to cuda
    csv_map = OrderedDict({"filename":[],"probability":[]})
    model.cuda()
    model.eval()
    #  打开一个 JSON 文件，用于保存最终的提交结果。
    with open("./submit/test_result.json","w",encoding="utf-8") as f :
        submit_results = []
        for i,(input,filepath) in enumerate(tqdm(test_loader)):
            #3.2 change everything to cuda and get only basename
            filepath = [os.path.basename(x) for x in filepath] 
            with torch.no_grad():
                image_var = Variable(input).cuda() 
                y_pred = model(image_var)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            #3.4 save probability to csv files
            csv_map["filename"].extend(filepath)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x : [float(i) for i in x.split(";")])
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            if pred_label > 43:
                pred_label = pred_label + 2
            submit_results.append({"image_id":row['filename'],"disease_class":pred_label})
        json.dump(submit_results,f,ensure_ascii=False,cls = MyEncoder)

#4. more details to build main function    
def main():
    fold = 1
    #4.1 mkdirs
  
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       
    #4.2 get model and optimizer
    model = get_net()
    model = torch.nn.DataParallel(model) 
    device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu') 
    
    model.to(device)
   
    optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    log = Logger() 
    log.open(config.logs + "log_train.txt",mode="a") 
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    #4.3 some parameters for  K-fold and restart model
    start_epoch = 0 
    best_precision1 = 0 
    best_precision_save = 0 
    resume = True
    
    #4.4 restart the training process
    # Resume training from checkpoint
    if resume:
        checkpoint = torch.load(config.best_models  + os.sep  +config.model_name + os.sep + str(fold) + "/model_best.pth.tar") 
        start_epoch = checkpoint["epoch"] 
        fold = checkpoint["fold"] 
        best_precision1 = checkpoint["best_precision1"] 
        model.load_state_dict(checkpoint["state_dict"]) 
        optimizer.load_state_dict(checkpoint["optimizer"]) 

    #4.5 get files and split for K-fold dataset
    #4.5.1 read files
    train_ = get_files(config.train_data,"train") 
    #val_data_list = get_files(config.val_data,"val")
    test_files = get_files(config.test_data,"test") 

    #4.5.2 and 4.5.3 Divide training set and validation set
    # Divide the training data into a training set and a validation set, and perform stratified sampling by label.
    train_data_list,val_data_list = train_test_split(train_,test_size = 0.15,stratify=train_["label"])
    #4.5.4 load dataset
    train_dataloader = DataLoader(ChaojieDataset(train_data_list),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=True)
    val_dataloader = DataLoader(ChaojieDataset(val_data_list,train=False),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=False)
    test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
    # Create a learning rate scheduler and multiply the learning rate by 0.1 every 10 epochs.
    scheduler =  optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma=0.1)
    #4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top2 = AverageMeter()
    valid_loss = [np.inf,0,0] 
    model.train()
   
    log.write('** start training here! **\n')
    log.write('                           |------------ VALID -------------|----------- TRAIN -------------|------Accuracy------|------------|\n')
    log.write('lr       iter     epoch    | loss   top-1  top-2            | loss   top-1  top-2           |    Current Best    | time       |\n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')
    #4.5.5 train
    start = timer() 
    for epoch in range(start_epoch,config.epochs): 
        scheduler.step(epoch) 
        for iter,(input,target) in enumerate(train_dataloader):
            #4.5.5 switch to continue train process
            model.train()
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target)).long()).to(device)
            output = model(input)
            loss = criterion(output,target)

            precision1_train,precision2_train = accuracy(output,target,topk=(1,2))
            train_losses.update(loss.item(),input.size(0))  
            train_top1.update(precision1_train[0],input.size(0)) # Update the training Top-1 accuracy counter.
            train_top2.update(precision2_train[0],input.size(0)) 

            #backward
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
            lr = get_learning_rate(optimizer)
          
            print('\r',end='',flush=True)
            print('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f         | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (\
                         lr, iter/len(train_dataloader) + epoch, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_losses.avg, train_top1.avg, train_top2.avg,str(best_precision_save),
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)

        #evaluate
        lr = get_learning_rate(optimizer)
        #evaluate every half epoch
        valid_loss = evaluate(val_dataloader,model,criterion) # Call the evaluate function to evaluate the model's performance on the validation set
        is_best = valid_loss[1] > best_precision1 # Determine whether the Top-1 accuracy of the current validation set is better than the historical best.
        best_precision1 = max(valid_loss[1],best_precision1) # Update the best Top-1 accuracy.
        # Try converting best accuracy to numpy array.
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass
        # Save the model checkpoint, and if the current model is the best model, save it as the best model.
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_precision1":best_precision1,
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "valid_loss":valid_loss,
        },is_best,fold)
        # clear current line
        print("\r",end="",flush=True)
       
        log.write('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f          | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (\
                        lr, 0 + epoch, epoch,
                        valid_loss[0], valid_loss[1], valid_loss[2],
                        train_losses.avg,    train_top1.avg,    train_top2.avg, str(best_precision_save),
                        time_to_str((timer() - start),'min'))
                )
        log.write('\n')
        # Sleep briefly to prevent log writing from being written too frequently.
        time.sleep(0.01)

    # Checkpoint to load the best model.
    best_model = torch.load(config.best_models + os.sep+config.model_name+os.sep+ str(fold) +os.sep+ 'model_best.pth.tar')
    # Load the state dictionary of the best model.
    model.load_state_dict(best_model["state_dict"])
    # Call the test function for testing.
    test(test_dataloader,model,fold)

if __name__ =="__main__":
    main()





















