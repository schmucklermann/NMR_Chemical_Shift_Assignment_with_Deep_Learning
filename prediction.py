#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from random import randint
import random
import seaborn as sns
import bisect


# In[2]:


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


# In[3]:


# Define the linear regression model
class LinearRegression(nn.Module):
    
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1024, 2)
    
    def forward(self, x):
        return self.linear(x)



# Define a two-layer FNN model
class TwoLayerFNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create or load your 1D CNN model
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, padding=3)
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x.permute(0,2,1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.permute(0,2,1)
        x = self.fc(x)  # Take the last output of the convolutional layers
        return x


    


import matplotlib.gridspec as gridspec
class SeabornFig2Grid():
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or             isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()
    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])
    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)
        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])
    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)
    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()
    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

        
def get_contour_plot(h_y,h_yhat,n_y,n_yhat,model,epoch,learning_rate,hidden_dim,output,train_type):
     
    fig = plt.figure(figsize=(20,15))
    gs = gridspec.GridSpec(1, 2)
    sns.set(font_scale=1.5)

    #1H
    g = sns.jointplot(x=h_y, y=h_yhat, color="magenta",kind='reg',scatter=False, xlim = (-4,4), ylim = (-4,4))
    r_h, p_h = stats.pearsonr(h_y, h_yhat)
    g.ax_joint.annotate(f'pearsonr = {r_h:.2f}, p = {p_h:.2f}',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',fontsize=35)
    g.plot_joint(sns.kdeplot, color="magenta", fill=True,bw_adjust=.5)
    g.set_axis_labels(xlabel='Groundtruth of 1H',ylabel='Prediction of 1H')
      

    #15N
    g1= sns.jointplot(x=n_y, y=n_yhat, color="blue", kind='reg',scatter=False, xlim = (-4,4), ylim = (-4,4))
    r_n, p_n = stats.pearsonr(n_y, n_yhat)
    g1.ax_joint.annotate(f'pearsonr = {r_n:.2f}, p = {p_n:.2f}',
                    xy=(0.1, 0.9), xycoords='axes fraction',
                    ha='left', va='center',fontsize=35)
    g1.plot_joint(sns.kdeplot, color="blue", fill=True,bw_adjust=.5)
    g1.set_axis_labels(xlabel='Groundtruth of 15N',ylabel='Prediction of 15N')
    
    mg0 = SeabornFig2Grid(g, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])

    gs.tight_layout(fig)
    gs.update(top=0.95)
    #plt.tight_layout()
    plt.savefig(output+"pearson_"+model+'_'+str(epoch)+'_'+str(learning_rate)+'_'+str(hidden_dim)+'_'+train_type+'.png',bbox_inches='tight')
    plt.show()
    plt.close()
    
    return (r_h, p_h,r_n, p_n)


# In[5]:


def plot_precision_coverage(precisions, coverages, step, model,num_epochs,learning_rate,hidden_dim,output,train_type,title="Coverage and Accuracy",include_acc=False):
    plt.figure(figsize=(8, 6))
    if include_acc:
        colors = [plt.get_cmap('viridis')(0.2), plt.get_cmap('viridis')(0.5), plt.get_cmap('viridis')(0.8)]
    else:
        colors = [plt.get_cmap('viridis')(0.2), plt.get_cmap('viridis')(0.8)]
    plt.plot(step, coverages, label='Coverage', color=colors[0], linewidth=2.5)
    plt.plot(step, precisions, label='Accuracy', color=colors[1], linewidth=2.5)
    if include_acc:
        plt.plot([x for x in [x * step for x in range(int(1 / step) + 1)]], accuracies, label='Accuracy', color=colors[2], linewidth=2.5)
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('%', fontsize=14)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend(fontsize=12)
    plt.xlim(max(step)+0.05, min(step)-0.05)
    plt.title(title, fontsize=18, pad=15)
    
    plt.savefig(output+"coverage_accuracy_"+model+'_'+str(num_epochs)+'_'+str(learning_rate)+'_'+str(hidden_dim)+'_'+train_type+'.png',bbox_inches='tight')
    plt.show()
    plt.close()

    


# In[6]:


def get_boxplot_accuracy(acc, acc_rdnm,model,num_epochs,learning_rate,hidden_dim,output,train_type):

    data = [acc, acc_rdnm]

    fig = plt.figure(figsize =(5, 7))
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(data,patch_artist=True)

    colors = ['darkmagenta','deepskyblue']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title("Accuracy per Protein")
    ax.set_xticklabels(['Proteins', 'Random'])

    # show plot
    plt.savefig(output+"accuracy_"+model+'_'+str(num_epochs)+'_'+str(learning_rate)+'_'+str(hidden_dim)+'_'+train_type+'.png',bbox_inches='tight')
    plt.show()
    plt.close()


# In[7]:


def plot_simple(x1, x2, label1, label2, x_label, y_label, title, x_range, output, out_name):

    plt.figure(figsize=(14, 6))
    
    l = range(0, len(x1))
    plt.plot(l, x1, label=label1)
    plt.plot(l, x2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    x_labels = np.arange(min(l), max(l)+1, x_range)
    plt.xticks(x_labels)
    plt.ylim(ymin=0)
    
    
    plt.legend()

    plt.savefig(output+out_name)
    plt.show()
    plt.close()
    
    
    
def plot_double(x1, x2, y1, y2, xlabel1, xlabel2, ylabel1, ylabel2, x_label, y_label, title, 
                x_range, y_range, output, out_name):
    

    plt.figure(figsize=(14, 9))
    
    l = range(0, len(x1))
    m1 = max(y1)
    m2 = max(y2)
    m = [m1,m2]
    
    plt.plot(l, y1, label=ylabel1)
    plt.plot(l, y2, label=ylabel2)
    plt.plot(l, x1, label=xlabel1)
    plt.plot(l, x2, label=xlabel2)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    x_labels = np.arange(min(l), max(l)+1, x_range)
    y_labels = np.arange(0, max(m)+0.1, y_range)
    
    plt.xticks(x_labels)
    plt.yticks(y_labels)
    plt.ylim(ymin=0)
    
    
    plt.legend()
    
    plt.savefig(output+out_name)
    plt.show()
    plt.close()


# In[8]:


from torch.nn.utils.rnn import pad_sequence

class MyCollator(object):
    
    def __call__(self, batch, ignore_idx=-100):
        # batch is a list of the samples returned by your __get_item__ method in your CustomDataset
        ids, X, Y = zip(*batch)
        X = pad_sequence(X, batch_first=True, padding_value=ignore_idx)
        Y = pad_sequence(Y, batch_first=True, padding_value=ignore_idx)
        return (list(ids), X, Y)

    
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, samples, first):
        
        #self.ids = []
        #self.inputs = []
        #self.targets = []
        
        self.item = []
        self.seq = []
        self.first = first
        
        for seq, item in samples.items():
            self.seq.append(seq)
            self.item.append(item)
            
        self.data_len = len(self.item)    
        
        
        #self.ids, self.inputs, self.targets = zip(*[(ids, inputs, targets)  
        #                    for ids, (inputs, targets) in samples.items()])
        #self.data_len = len(self.inputs) # number of samples in the set
        
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        
        curr_item = self.item[index]
        
        if self.first:
            i = 0
        else:
            length = len(curr_item)
            i = randint(0, length-1)
            
        item = curr_item[i]
        ids = item[0]
        x = item[1]
        y = item[2]    
            
        
        #ids = self.ids[index]
        #x = self.inputs[index]#.float()
        #y = self.targets[index]#.long()
        return (ids, torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    
def get_dataloader(customdata, batch_size, first):
    # Create dataloaders with collate function
    my_collator = MyCollator()
    dataset = CustomDataset(customdata, first)
    return torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        drop_last=False,
                                        collate_fn=my_collator)
  

class EarlyStopper():
    def __init__(self, log_dir, model_choice, input_dim, hidden_dim, output_dim):
        self.log_dir      = log_dir
        self.checkpoint_p = log_dir / 'checkpoint.pt'
        self.epsilon      = 1e-3 # the minimal difference in improvement a model needs to reach
        self.min_loss     = np.Inf # counter for lowest/best overall-loss
        self.n_worse      = 0 # counter of consecutive non-improving losses
        self.patience     = 5 # number of max. epochs accepted for non-improving loss
        self.model_choice = model_choice
        self.input_dim    = input_dim 
        self.hidden_dim   = hidden_dim
        self.output_dim   = output_dim
       
    
    def load_checkpoint(self):
        state = torch.load( self.checkpoint_p)
        model = get_model(model_choice=self.model_choice, input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim=self.output_dim)
        model.load_state_dict(state['state_dict'])
        print('Loaded model from epoch: {:.1f}'.format(state['epoch']))
        return model, state['epoch']
    
    def save_checkpoint(self, model, epoch, optimizer):
        state = { 
                    'epoch'      : epoch,
                    'state_dict' : model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                }
        torch.save( state, self.checkpoint_p )
        return None
    

         
        
    def check_performance(self, model, test_loader, crit, optimizer, epoch, num_epochs):
        current_loss,v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage, precision,bins, hn = testing(model, test_loader, crit, epoch, num_epochs, set_name="VALID")
    
        # if the model improved compared to previously best checkpoint
        if current_loss < (self.min_loss - self.epsilon):
            print('New best model found with loss= {:.3f}'.format(current_loss))  
            self.save_checkpoint( model, epoch, optimizer)
            self.min_loss = current_loss # save new loss as best checkpoint
            self.n_worse  = 0
        else: # if the model did not improve further increase counter
            self.n_worse += 1
            if self.n_worse > self.patience: # if the model did not improve for 'patience' epochs
                print('Stopping due to early stopping after epoch {}!'.format(epoch))
                return True, current_loss, v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage, precision,bins, hn
        return False, current_loss,v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage, precision, bins, hn

    




#Mask residues with -100, replace with 0     
def mask(Y, Yhat):
       
    true_false = Y!=-100
    masked_y = torch.where(true_false,Y, 0)
    masked_yhat = torch.where(true_false, Yhat, 0)
    
    return masked_y, masked_yhat     


#calcluate Loss for H and N individually
def get_loss(Y, Yhat, crit):
    
    y_masked_h, yhat_masked_h = mask(Y[:,:,0],  Yhat[:,:,0])
    y_masked_n, yhat_masked_n = mask(Y[:,:,1],  Yhat[:,:,1])
    
    
    loss_h = crit(y_masked_h, yhat_masked_h)
    loss_n = crit(y_masked_n, yhat_masked_n)
    
    return loss_h, loss_n   


        
def training(model, trainloader, crit, optimizer, epoch, num_epochs):
    
    model.train() # ensure model is in training mode (dropout, batch norm, ...)
    accuracies = []
    batch = 0
    batch_loss = 0
    batch_loss_h = 0
    batch_loss_n = 0
    
    #start = time.time()
 
    for i, (_, X, Y) in enumerate(trainloader): # iterate over all mini-batches in train
      
        optimizer.zero_grad() # zeroes the gradient buffers of all parameters
        
        X    = X.to(device)
        Y    = Y.to(device)
        
        #train
        Yhat = model(X)
        
        #loss
        #Loss normalisation
        #-loss konstanter Faktor multiplikation
        #-raw input data normalization (zscore)
        #total_loss = (loss_h + loss_n)/2 #mean(loss) oder selber effekt: nur summe(loss) dafür learning rate runter
    
        loss_h, loss_n = get_loss(Y, Yhat, crit)
        total_loss = (loss_h + loss_n)/2
        
        #backpropagation
        total_loss.backward() 
        optimizer.step()
        
        
        
        #sum up loss per batch
        batch += 1
        batch_loss += total_loss.detach().cpu().numpy()
        batch_loss_h += loss_h.detach().cpu().numpy()
        batch_loss_n += loss_n.detach().cpu().numpy()
        
        
    #collect loss for plot: mean over batches per epoch
    epoch_loss = batch_loss/batch
    epoch_loss_h = batch_loss_h/batch
    epoch_loss_n = batch_loss_n/batch
        
    #end = time.time()
    if epoch % 1 == 0 or epoch == num_epochs:
        out = ('Epoch [{}/{}], TRAIN loss: {:.2f}').format( 
                    epoch, num_epochs, 
                    epoch_loss,
                    #end-start,
                    )
        print(out)
        
    return epoch_loss,epoch_loss_h,epoch_loss_n 



def testing( model, testloader, crit, epoch, num_epochs, log_dir=None, set_name=None):
    model.eval() # [ensure model is in training mode (dropout, batch norm, ...)
    accuracies = []
    accuracies_rndm = []
    
    
    h_y = []
    h_yhat = []
    n_y = []
    n_yhat = []
    
    auc_list = []
    coverage_all = []
    precision_all = []
    
    distances_all = [] #list of distances to 1-NN
    truepredictions = [] #correctly labeled 1-NNs
    prot_len_list = []
    

    #start = time.time()
    results = {}
    
    batch = 0
    batch_loss = 0
    batch_loss_h = 0
    batch_loss_n = 0
    
    for i, (pdb_ids, X, Y) in enumerate(testloader):
        
        # IN: [B, L, F] OUT: [B, N, L]
        X    = X.to(device)
        Y    = Y.to(device)


        with torch.no_grad():
            Yhat = model(X)
   
        loss_h, loss_n = get_loss(Y, Yhat, crit)
        total_loss = (loss_h + loss_n)/2
        
        
        # sum up loss per batch
        batch += 1
        batch_loss += total_loss.detach().cpu().numpy()
        batch_loss_h += loss_h.detach().cpu().numpy()
        batch_loss_n += loss_n.detach().cpu().numpy()
        
        
        # IN: [B, N, L] OUT: [B, L]
        # iterate over every sample in batch
        for sample_idx in range(0,Yhat.shape[0]):
            
            yhat = Yhat[sample_idx] # get single sample/protein from mini-batch
            y    = Y[sample_idx]
            
            #Mask vectors
            y_masked_h, yhat_masked_h = mask(y[:,0],  yhat[:,0])
            y_masked_n, yhat_masked_n = mask(y[:,1],  yhat[:,1])
            
            y_masked_h = y_masked_h[y_masked_h.nonzero().squeeze()]
            y_masked_n = y_masked_n[y_masked_n.nonzero().squeeze()]
            yhat_masked_h = yhat_masked_h[yhat_masked_h.nonzero().squeeze()]
            yhat_masked_n = yhat_masked_n[yhat_masked_n.nonzero().squeeze()]
            
            y_masked = torch.stack((y_masked_h, y_masked_n),-1)
            yhat_masked = torch.stack((yhat_masked_h, yhat_masked_n),-1)
            
            
            #Collect all values for pearson per epoch
            h_y.extend(y_masked_h.tolist())
            h_yhat.extend(yhat_masked_h.tolist())
            n_y.extend(y_masked_n.tolist())
            n_yhat.extend(yhat_masked_n.tolist())
            
            #Classification Accuracy
            #nearest neighbor: right/wrong assignment
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(y_masked.cpu()) 
            distances, indices = nbrs.kneighbors(yhat_masked.cpu())
            
            
            #collect true prediction, if kNN matches position
            indices = indices.flatten()
            prot_len = len(indices)
            index_comparison = np.arange(prot_len)
            
        
            
            #acc = correct predictions/number of predictions
            comp = (indices==index_comparison)
            true = (comp == True).sum()
            acc = true/prot_len
            accuracies.append(acc)
            
            
            
            #random index list to compare accuracy against
            res = random.sample(range(0, prot_len),prot_len)
            comp_rndm = (res==index_comparison)
            true_rndm = (comp_rndm == True).sum()
            acc_rndm = true_rndm/prot_len
            accuracies_rndm.append(acc_rndm)
            
           
        
            #Precision, Coverage
            #Coverage: wie viele meiner Residues kann ich bei gegebenem Distance-cut-off X noch vorhersagen 
            #y% aller Residues haben Distance<X, zB 30% aller Residues haben eine Distanz<0.4 zum nearest neighbor
            #Precision: bei gegbenem Distance-cut-off X, sind wie viele deiner vorhergesagten Residues korrekt 
            #z% der Residues mit Distance<X sind korrekt., zB 10% der Residues die bei Distanz<0.4 vorhergesagt werden sind korrekt.
            
            distances = distances.flatten()
            
            distances_all.extend(distances)
            
            truepredictions.extend(comp)
            p = np.full(len(distances), prot_len)
            prot_len_list.extend(p)
            
            
            if epoch==num_epochs: # store predictions of final checkpoint for writing to log
                pdb_id = pdb_ids[sample_idx]
                results[pdb_id] = (','.join([str(i.detach().cpu().numpy()) for i in y]),
                                   ','.join([str(j.detach().cpu().numpy()) for j in yhat])
                                   ,accuracies[-1])
    
    
    
    
    #Coverage/Precision (Accuracy)
    
    #sort distances
    sorted_lists = sorted(zip(distances_all, truepredictions), key=lambda x: x[0], reverse=True)
    dist, trues = zip(*sorted_lists)
    
    max_dist = dist[0]
    #min_dist = dist[-1]
    
    bins = np.arange(0, max_dist + 0.01, 0.01)[::-1]
    set_len = len(dist)
    
   
    for b in bins:
        
        start_index = next((index for index, value in enumerate(dist) if value < b), set_len)
        n = len(dist[start_index:])

        # calculate coverage/precision
        if n == 0:
            prec = 0
            cov = 0
        else:
            # Use list comprehension to count True values within the range of the first bin
            trues_in_bin = sum(trues[start_index:])
            
            prec = trues_in_bin/n
            cov = n/set_len
        

        precision_all.append(prec)
        coverage_all.append(cov)
        
    
    
    #Accuracy Mean
    acc_mean = sum(accuracies)/len(accuracies)
    acc_mean_rndm = sum(accuracies_rndm)/len(accuracies_rndm)

    
    #collect loss for plot: mean over batches per epoch
    epoch_loss = batch_loss/batch
    epoch_loss_h = batch_loss_h/batch
    epoch_loss_n = batch_loss_n/batch
            
    #end = time.time()
    if epoch % 1 == 0 or epoch == num_epochs:
        out = ('Epoch [{}/{}], {} loss: {:.2f}, Accuracy Mean: {:.2f}, Accuracy RNDM: {:.2f}').format( 
                    epoch,num_epochs, set_name,
                    epoch_loss,
                    acc_mean, acc_mean_rndm
                    #end-start,
                    )
        print(out)
        
    if epoch==num_epochs:
        write_predictions(results, log_dir, set_name)
    
    

  
    return epoch_loss,epoch_loss_h,epoch_loss_n, accuracies, accuracies_rndm, coverage_all, precision_all, bins, (h_y,h_yhat,n_y,n_yhat) 





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_predictions(results, log_dir, set_name):
    out_p = log_dir / (set_name + '_log.txt')
    with open(out_p, 'w+') as out_f:
        out_f.write('\n'.join( 
            ">{},y,yhat,acc={:.3f}\n{}\n{}".format(pdb_id, acc, y, yhat) 
                              for pdb_id, (y,yhat,acc) in results.items() ) )
    return None




def get_model(model_choice,input_dim, hidden_dim, output_dim):
     if model_choice == "CNN":
        return CNN(input_dim, output_dim).to(device)
     elif model_choice == "FNN":
        return TwoLayerFNN(input_dim, hidden_dim, output_dim).to(device)
     elif model_choice == "LinReg":
        return LinearRegression().to(device)
     else:
        raise NotImplementedError

        
        
#Get Loss of Epoch 0        
def get_initialized_loss(model, dataloader, crit):
    batch = 0
    batch_loss = 0
    batch_loss_h = 0
    batch_loss_n = 0
    
    for i, (pdb_ids, X, Y) in enumerate(dataloader):
        
        X    = X.to(device)
        Y    = Y.to(device)
        
        with torch.no_grad():
            Yhat = model(X)

        loss_h, loss_n = get_loss(Y, Yhat, crit)
        total_loss = (loss_h + loss_n)/2
        
        batch += 1
        batch_loss += total_loss.detach().cpu().numpy()
        batch_loss_h += loss_h.detach().cpu().numpy()
        batch_loss_n += loss_n.detach().cpu().numpy()


    epoch_loss = batch_loss/batch
    epoch_loss_h = batch_loss_h/batch
    epoch_loss_n = batch_loss_n/batch
    
    return epoch_loss,epoch_loss_h,epoch_loss_n


# In[9]:


def predict(model_choice, input_dim, hidden_dim, output_dim,batch_size,learning_rate, num_epochs,train,valid,test, output, train_type):

    train_loader = get_dataloader(train,batch_size=batch_size,first=True)
    val_loader = get_dataloader(valid,batch_size=batch_size,first=True)
    test_loader = get_dataloader(test,batch_size=batch_size,first=True)
    

    #val_aa_mean = get_dataloader(valid_aa_mean,batch_size=batch_size,first=True)


    root = Path.cwd() 
    # create log directory if it does not exist yet
    log_root = root / "log"
    if not log_root.is_dir():
        print("Creating new log-directory: {}".format(log_root))
        log_root.mkdir()

    log_dir = log_root



    model = get_model(model_choice, input_dim, hidden_dim, output_dim)

    early_stopper = EarlyStopper(log_dir, model_choice, input_dim, hidden_dim, output_dim)

    n_free_paras = count_parameters(model)
    print('Number of free parameters: {}'.format( n_free_paras))

    crit = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)


    #Loss
    train_loss = []
    train_loss_h = []
    train_loss_n = []

    valid_loss = []
    valid_loss_h = []
    valid_loss_n = []

    

    #Initial Loss of Epoch 0
    #for Loss Plot: take mean per epoch of all batches
    #for Loss propagation: take every single loss per batch
    t_epoch_loss,t_epoch_loss_h,t_epoch_loss_n = get_initialized_loss(model, train_loader,crit)
    v_epoch_loss,v_epoch_loss_h,v_epoch_loss_n = get_initialized_loss(model, val_loader, crit)

    train_loss.append(math.sqrt(t_epoch_loss))
    train_loss_h.append(math.sqrt(t_epoch_loss_h))
    train_loss_n.append(math.sqrt(t_epoch_loss_n))

    valid_loss.append(math.sqrt(v_epoch_loss))
    valid_loss_h.append(math.sqrt(v_epoch_loss_h))
    valid_loss_n.append(math.sqrt(v_epoch_loss_n))


    #TODO: Log datei 
    
    print(train_type)

    #start = time.time()
    for epoch in range(num_epochs): # for each epoch: train & valid
        stop, v_epoch_loss,v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage, precision,bins, hn = early_stopper.check_performance(model, val_loader, crit, optimizer, epoch, num_epochs)
        if stop: # if early stopping criterion was reached
            break
        t_epoch_loss,t_epoch_loss_h,t_epoch_loss_n  = training(model, train_loader, crit, optimizer, epoch, 
                                                               num_epochs) 
        """v_epoch_loss,v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage, precision, hn = testing(model, 
                                                                                val_loader, crit, 
                                                                                epoch, num_epochs, 
                                                                                log_dir=log_dir, set_name="VAL")"""



        # Collect loss per epoch
        train_loss.append(math.sqrt(t_epoch_loss))
        train_loss_h.append(math.sqrt(t_epoch_loss_h))
        train_loss_n.append(math.sqrt(t_epoch_loss_n))

        valid_loss.append(math.sqrt(v_epoch_loss))
        valid_loss_h.append(math.sqrt(v_epoch_loss_h))
        valid_loss_n.append(math.sqrt(v_epoch_loss_n))

        


        #Random Baseline: H/N mean per AA
        """v_epoch_loss,v_epoch_loss_h,v_epoch_loss_n, acc, acc_rndm, coverage_all, precision_all, hn = testing(model, 
                                                                                val_aa_mean, crit, 
                                                                                epoch, num_epochs, 
                                                                                log_dir=log_dir, set_name="VAL_AA_Mean")
        acc_valid_aa_mean.extend(acc)"""


    # load the model weights of the best checkpoint
    model = early_stopper.load_checkpoint()[0]
    #end = time.time()
    #print('Total training time: {}[m]'.format((end-start)/60))
    print('Running final evaluation on the best checkpoint.')
    _,_,_, acc, acc_rndm, coverage, precision,bins, hn = testing(model,test_loader, crit, epoch, num_epochs, 
                                                                                log_dir=log_dir, set_name="Test")

    
    

    #Plots
    #Precision/Coverage
    #get_plot_precision_coverage(coverage,precision,model_choice,num_epochs,learning_rate,hidden_dim,output,train_type)
    plot_precision_coverage(precision, coverage, bins, model_choice, num_epochs,learning_rate,hidden_dim,output,train_type,title="Coverage and Accuracy",include_acc=False)
    
    #Accuracy
    #get_boxplot_accuracy(acc_valid, acc_rdnm_valid, acc_valid_aa_mean, model_choice,num_epochs,learning_rate,hidden_dim,output,train_type)
    get_boxplot_accuracy(acc, acc_rndm, model_choice,num_epochs,learning_rate,hidden_dim,output,train_type)
    
    
    
    #Pearson correlation: predicted vs groundtruth
    pearson = get_contour_plot(hn[0],hn[1],hn[2],hn[3], model_choice,epoch,learning_rate,hidden_dim,output,train_type)
    
    #Loss
    t = 'Training and Validation Loss ('+train_type+', '+model_choice+')'
    out_loss = 'loss_'+train_type+'_'+str(num_epochs)+'_'+str(learning_rate)+'_'+str(hidden_dim)+'.png'
    out_losshn = 'lossHN_'+train_type+'_'+str(num_epochs)+'_'+str(learning_rate)+'_'+str(hidden_dim)+'.png'
    
    
    plot_simple(train_loss, valid_loss, 'Training Loss', 'Validation Loss', 'Epochs', 'Loss', 
            t, 1, output, out_loss)
    
    #Loss seperate H/N
    plot_double(train_loss_h,valid_loss_h,train_loss_n, valid_loss_n, 
            'Training Loss H', 'Training Loss N', 
            'Validation Loss H','Validation Loss N',
            'Epochs', 'Loss',t, 
            1, 0.05, output,out_losshn)
    
    return acc, acc_rndm, pearson


# In[10]:


"""# Hyper parameters
num_epochs    = 1
learning_rate = 1e-3
batch_size    = 128 

# Initialize the model
input_dim = 1024  # Number of input features
output_dim = 2  # Number of output values
hidden_dim = 513  # Number of neurons in the hidden layer: (input_dim + output_dim) / 2,

model_choice = "FNN" 

output = "./"

predict(model_choice, input_dim, hidden_dim, output_dim,batch_size,learning_rate,
        num_epochs,train,test,valid, output)"""

