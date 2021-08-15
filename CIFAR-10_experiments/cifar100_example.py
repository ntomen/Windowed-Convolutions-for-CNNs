# -*- coding: utf-8 -*-
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from windowed_conv import Conv2d_window
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description='CIFAR100 training with 7x7 window in all layers and weight decay')
parser.add_argument('--seed', default=None, type=int, help='rng seed')
parser.add_argument('--n_layers', default=6, type=int, help='total number of conv layers')
parser.add_argument('--save', type=str, default='/save_dir')

args = parser.parse_args()

# Fix seed
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')
#-----------------------------------------------------------------------------------------------
# Other useful definitions
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape=torch.prod(torch.tensor(x.shape[1:])).item()
        return x.reshape(-1,shape) # batchsize-by-rest

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs(args.save)

CELoss=nn.CrossEntropyLoss()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#-----------------------------------------------------------------------------------------------
# Define CIFAR100 dataloaders
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ToTensor()])
transform_test=transforms.Compose([transforms.ToTensor()])
data_path='/path_to_cifar100'
batch_size=32

# Training data
cifar100_trainset=datasets.CIFAR100(root=data_path, train=True, transform=transform_train, download=True)
train_dl=torch.utils.data.DataLoader(cifar100_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Validation data
cifar100_testset=datasets.CIFAR100(root=data_path, train=False, transform=transform_test, download=True)
test_dl=torch.utils.data.DataLoader(cifar100_testset, batch_size=batch_size, shuffle=False, num_workers=2)
#-----------------------------------------------------------------------------------------------
# Define the 2 models
n_inp_ch=3 # CIFAR100
n_out_ch=128
n_out2_ch=256
n_classes=100 # CIFAR100

# Define networks
n_layers=args.n_layers

block=[]
block.append(torch.nn.Conv2d(n_inp_ch,n_out_ch,kernel_size=(7,7),stride=2,padding=(3,3),bias=False))
block.append(torch.nn.BatchNorm2d(n_out_ch))
block.append(torch.nn.ReLU())
block.append(torch.nn.Conv2d(n_out_ch,n_out2_ch,kernel_size=(7,7),padding=(3,3),bias=False))
block.append(torch.nn.BatchNorm2d(n_out2_ch))
block.append(torch.nn.ReLU())

block_win=[]
block_win.append(Conv2d_window(n_inp_ch,n_out_ch,kernel_size=(7,7),stride=2,padding=(3,3),bias=False))
block_win.append(torch.nn.BatchNorm2d(n_out_ch))
block_win.append(torch.nn.ReLU())
block_win.append(Conv2d_window(n_out_ch,n_out2_ch,kernel_size=(7,7),padding=(3,3),bias=False))
block_win.append(torch.nn.BatchNorm2d(n_out2_ch))
block_win.append(torch.nn.ReLU())

if n_layers>2:
    for i in range(n_layers-2):
        block.append(torch.nn.Conv2d(n_out2_ch,n_out2_ch,kernel_size=(7,7),padding=(3,3),bias=False))
        block.append(torch.nn.BatchNorm2d(n_out2_ch))
        block.append(torch.nn.ReLU())

        block_win.append(Conv2d_window(n_out2_ch,n_out2_ch,kernel_size=(7,7),padding=(3,3),bias=False))
        block_win.append(torch.nn.BatchNorm2d(n_out2_ch))
        block_win.append(torch.nn.ReLU())

model=torch.nn.Sequential(*block,torch.nn.AdaptiveAvgPool2d((1,1)),\
               Flatten(),torch.nn.Linear(n_out2_ch,n_classes)).to(device)

model_win=torch.nn.Sequential(*block_win,torch.nn.AdaptiveAvgPool2d((1,1)),\
               Flatten(),torch.nn.Linear(n_out2_ch,n_classes)).to(device)

if n_layers>13:
    warnings.warn("Networks deeper than 13 layers may not be ideal for small image sizes "+\
             +"(e.g. 32 x 32) and with the current width/pooling specifications")
#-----------------------------------------------------------------------------------------------
# Define optimizer
optim=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9, weight_decay=0.001)
optim_win=torch.optim.SGD(model_win.parameters(),lr=0.01,momentum=0.9, weight_decay=0.001)
#-----------------------------------------------------------------------------------------------
# Training
loss_counter=0
log_loss=20 # track loss every 20 batches
n_epoch=150
n_train_img=len(cifar100_trainset)

plot_loss=np.zeros(n_epoch*int(np.floor(np.ceil(n_train_img/batch_size)/log_loss)))
plot_loss_win=np.zeros(n_epoch*int(np.floor(np.ceil(n_train_img/batch_size)/log_loss)))

for i_epoch in range(n_epoch):
    print(i_epoch)
    batch_counter=0
    running_loss=0
    running_loss_win=0

    # Learning rate decay
    if i_epoch==75 or i_epoch==120:
        for param_group in optim.param_groups:
            param_group['lr']=0.1*param_group['lr']
        for param_group in optim_win.param_groups:
            param_group['lr']=0.1*param_group['lr']

    for X,y in train_dl:
        batch_counter+=1
        # Train
        X=X.to(device)
        y=y.to(device)

        # Backprop model 1
        optim.zero_grad()
        pred=model(X)
        loss=CELoss(pred,y)
        loss.backward()
        optim.step()

        # Backprop model 2
        optim_win.zero_grad()
        pred_win=model_win(X)
        loss_win=CELoss(pred_win,y)
        loss_win.backward()
        optim_win.step()

        running_loss+=loss.item()
        running_loss_win+=loss_win.item()

        if batch_counter%log_loss==0:
            # print things
            print(batch_counter)
            print(np.argmax(pred[0,:].detach().cpu().numpy()),\
                  np.argmax(pred_win[0,:].detach().cpu().numpy()),\
                  y[0].detach().cpu().numpy())

            # update loss trace
            plot_loss[loss_counter]=running_loss
            plot_loss_win[loss_counter]=running_loss_win
            loss_counter+=1

            running_loss=0
            running_loss_win=0

# Plot losses
plt.figure(1)
plt.plot(plot_loss,label='conv')
plt.plot(plot_loss_win,label='conv_win')
plt.legend()
plt.savefig(os.path.join(args.save, 'loss_cifar100_all_win_wd_'+str(n_layers)+'_layers_'+str(args.seed)+'.pdf'), bbox_inches='tight')

# Plot losses zoomed in
plt.figure(2)
plt.plot(plot_loss[-100:],label='conv')
plt.plot(plot_loss_win[-100:],label='conv_win')
plt.legend()
plt.savefig(os.path.join(args.save, 'loss_zoom_cifar100_all_win_wd_'+str(n_layers)+'_layers_'+str(args.seed)+'.pdf'), bbox_inches='tight')
#-----------------------------------------------------------------------------------------------
# Validation
accuracy=np.zeros(n_classes)
accuracy_win=np.zeros(n_classes)
class_counter=np.zeros(n_classes)

loss_counter=0
log_loss=20 # track loss every 20 batches
n_test_img=len(cifar100_testset)

plot_val_loss=np.zeros(int(np.floor(np.ceil(n_test_img/batch_size)/log_loss)))
plot_val_loss_win=np.zeros(int(np.floor(np.ceil(n_test_img/batch_size)/log_loss)))

print('Validating...')

batch_counter=0
running_loss=0
running_loss_win=0

with torch.no_grad():
    model.eval()
    model_win.eval()

    for X,y in test_dl:
        # Test
        batch_counter+=1
        X=X.to(device)

        pred=model(X)
        pred_win=model_win(X)

        loss=CELoss(pred,y.to(device))
        loss_win=CELoss(pred_win,y.to(device))
        running_loss+=loss.item()
        running_loss_win+=loss_win.item()

        # Get accuracy
        for j in range(int(y.shape[0])):
            i_class=y[j].detach().numpy().item()

            class_counter[i_class]+=1
            accuracy[i_class]+=(i_class==np.argmax(pred[j,:].detach().cpu().numpy()))
            accuracy_win[i_class]+=(i_class==np.argmax(pred_win[j,:].detach().cpu().numpy()))

        if batch_counter%log_loss==0:
            # print things
            print(batch_counter)

            # update loss trace
            plot_val_loss[loss_counter]=running_loss
            plot_val_loss_win[loss_counter]=running_loss_win
            loss_counter+=1

            running_loss=0
            running_loss_win=0

# Plot val loss
plt.figure(3)
plt.plot(plot_val_loss,label='conv')
plt.plot(plot_val_loss_win,label='conv_win')
plt.legend()
plt.savefig(os.path.join(args.save, 'val_loss_cifar100_all_win_wd_'+str(n_layers)+'_layers_'+str(args.seed)+'.pdf'), bbox_inches='tight')

# Print accuracies
print(accuracy)
print(accuracy_win)
print(class_counter)
print(accuracy.sum()/class_counter.sum()*100)
print(accuracy_win.sum()/class_counter.sum()*100)

# Save model + losses
torch.save({'state_dict': model.state_dict(), 'state_dict_win': model_win.state_dict(), 'args': args, 'training_loss': plot_loss, 'training_loss_win': plot_loss_win, 'val_loss': plot_val_loss, 'val_loss_win': plot_val_loss_win, 'accuracy': accuracy, 'accuracy_win': accuracy_win, 'class_counter': class_counter}, os.path.join(args.save, 'model_cifar100_all_win_wd_'+str(n_layers)+'_layers_'+str(args.seed)+'.pth'))
