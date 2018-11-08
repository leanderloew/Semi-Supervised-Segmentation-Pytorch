import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
n_iter=0
def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
def IOU_Score(y_pred,y_val):
    def IoUOld(a,b):
        intersection = ((a==1) & (a==b)).sum()
        union = ((a==1) | (b==1)).sum()
        if union > 0:
            return intersection / union
        elif union == 0 and intersection == 0:
            return 1
        else:
            return 0
        
    y_pred=y_pred[:,1,:,:]#.view(batch_size,1,101,101)

    t=0.5
    IOU_list=[]
    for j in range(y_pred.shape[0]):
        y_pred_ = np.array(y_pred[j,:,:] > t, dtype=bool)
        y_val_=np.array(y_val[j,:,:], dtype=bool)

        IOU = IoUOld(y_pred_, y_val_) 

        IOU_list.append(IOU)
    #now we take different threshholds, these threshholds 
    #basically determine if our IOU consitutes as a "true positiv"
    #or not 
    prec_list=[]
    for IOU_t in np.arange(0.5, 1.0, 0.05):
        #get true positives, aka all examples where the IOU is larger than the threshhold
        TP=np.sum(np.asarray(IOU_list)>IOU_t)
        #calculate the current precision, by devididing by the total number of examples ( pretty sure this is correct :D)
        #they where writing the denominator as TP+FP+FN but that doesnt really make sens becasue there are no False postivies i think
        Prec=TP/len(IOU_list)
        prec_list.append(Prec)

    return np.mean(prec_list)



#Main Training Function 
from losses import lovasz_softmax,FocalLoss
from training_functions import IOU_Score
focal=FocalLoss(size_average=True)

def train(train_loader,segmentation_module,segmentation_ema,optimizer
          ,writer
          ,lovasz_scaling=0.1
          ,focal_scaling=0.9
          ,unsupervised_scaling=0.1
          ,ema_scaling=0.2
          ,non_ema_scaling=1
          ,second_batch_size=2
          ,train=True
          ,test=False
          ,writer_name_list=None
             ):
    
    global n_iter
    #Training Loop 
    cudnn.benchmark = True
    
    lovasz_scaling=torch.tensor(lovasz_scaling).float().cuda()
    focal_scaling=torch.tensor(focal_scaling).float().cuda()
    unsupervised_scaling=torch.tensor(unsupervised_scaling).float().cuda()
    ema_scaling=torch.tensor(ema_scaling).float().cuda()
    non_ema_scaling=torch.tensor(non_ema_scaling).float().cuda()
    
    #average meter for all the losses we keep track of. 
    ave_total_loss = AverageMeter() # Total Loss
    ave_non_ema_loss = AverageMeter()
    ave_ema_loss = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_lovasz_loss = AverageMeter()
    ave_focal_loss = AverageMeter()
    ave_lovasz_loss_ema = AverageMeter()
    ave_focal_loss_ema = AverageMeter()
    ave_unsupervised_loss = AverageMeter()
    ave_iou_score = AverageMeter()
    if train==True:
        segmentation_module.train()
        segmentation_ema.train()
    else:
        segmentation_module.eval()
        segmentation_ema.eval()
    
    for batch_data in train_loader:
        
        batch_data["img_data"]=batch_data["img_data"].cuda()
        batch_data["seg_label"]=batch_data["seg_label"].cuda().long().squeeze()

        #Normal Pred and Pred from the self ensembeled model 
        pred  = segmentation_module(batch_data)
        pred_ema  = segmentation_ema(batch_data)
        #We dont want to gradient descent into the EMA model 
        pred_ema=Variable(pred_ema.detach().data, requires_grad=False)

        ### UNSUPVERVISED LOSS ####
        unsupervised_loss = torch.mean((pred - pred_ema)**2).cuda()

        ###  SUPERVISED LOSS   ####
        #We jsut get rid of the Unlabeled examples for the supervised loss! 
        pred=pred[:-second_batch_size,:,:]
        pred_ema=pred_ema[:-second_batch_size,:,:]
        batch_data["seg_label"]=batch_data["seg_label"][:-second_batch_size,:,:]

        lovasz_loss=lovasz_softmax(pred, batch_data['seg_label'],ignore=-1,only_present=True).cuda()
        focal_loss=focal(pred, batch_data['seg_label'],)

        lovasz_loss_ema=lovasz_softmax(pred_ema, batch_data['seg_label'],ignore=-1,only_present=True).cuda()
        focal_loss_ema=focal(pred_ema, batch_data['seg_label'],)

        #### Loss Combinations #####
        non_ema_loss=(lovasz_loss*lovasz_scaling+focal_loss*focal_scaling).cuda()
        ema_loss=(lovasz_loss_ema*lovasz_scaling+focal_loss_ema*focal_scaling).cuda()

        total_loss=(non_ema_loss*non_ema_scaling+ema_loss*ema_scaling+unsupervised_scaling*unsupervised_loss).cuda()
        #Need to give it as softmaxes 
        pred = nn.functional.softmax(pred, dim=1)
        iou_score=IOU_Score(pred,batch_data["seg_label"])
        
        ### BW ####
        if train==True:
            optimizer.zero_grad()
            total_loss.backward()

            optimizer.step()  
            n_iter=n_iter+1
            
            
            update_ema_variables(segmentation_module, segmentation_ema, 0.999, n_iter)


        ### WRITING STUFF #########

        ave_non_ema_loss.update(non_ema_loss.data.item())
        ave_ema_loss.update(ema_loss.data.item())
        ave_total_loss.update(total_loss.data.item())
        ave_lovasz_loss.update(lovasz_loss.data.item())
        ave_focal_loss.update(focal_loss.data.item())
        ave_lovasz_loss_ema.update(lovasz_loss_ema.data.item())
        ave_focal_loss_ema.update(focal_loss_ema.data.item())
        ave_unsupervised_loss.update(unsupervised_loss.data.item())
        ave_iou_score.update(iou_score.item())
        
        if test==True:
            print(n_iter)
            break

        
        
    
    writer.add_scalar(writer_name_list[0], ave_non_ema_loss.average(), n_iter)        
    writer.add_scalar(writer_name_list[1], ave_ema_loss.average(), n_iter)
    writer.add_scalar(writer_name_list[2], ave_total_loss.average(), n_iter)        
    writer.add_scalar(writer_name_list[3], ave_lovasz_loss.average(), n_iter)  
    writer.add_scalar(writer_name_list[4], ave_focal_loss.average(), n_iter)        
    writer.add_scalar(writer_name_list[5], ave_lovasz_loss_ema.average(), n_iter)  
    writer.add_scalar(writer_name_list[6], ave_focal_loss_ema.average(), n_iter)        
    writer.add_scalar(writer_name_list[7], ave_unsupervised_loss.average(), n_iter)  
    writer.add_scalar(writer_name_list[8], ave_iou_score.average(), n_iter)
        
    if train==False:
        return np.mean(ave_iou_score.average())




