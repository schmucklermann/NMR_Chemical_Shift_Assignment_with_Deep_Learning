#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn

import trainingset_check as ts


# In[3]:


train_unfiltered_json = './data/Markus_trainsets/unfiltered.json'
train_unfiltered_tsv = './data/Markus_trainsets/unfiltered_rest_clu.tsv'

train_tolerant_json = './data/Markus_trainsets/tolerant.json'
train_tolerant_tsv = './data/Markus_trainsets/tolerant_rest_clu.tsv'

train_moderate_json = './data/Markus_trainsets/moderate.json'
train_moderate_tsv = './data/Markus_trainsets/moderate_rest_clu.tsv'

train_strict_json = './data/Markus_trainsets/strict.json'
train_strict_tsv = './data/Markus_trainsets/strict_rest_clu.tsv'

validation_fasta = './data/Markus_trainsets/rr_CheZOD117_test_set.fasta'
test_fasta = './data/Markus_trainsets/TriZOD_test_set.fasta'

h5_file = './data/BMRB_unfiltered_all.h5'

output = "./data/Markus_trainsets/Markus_trainsets_plots/"


# In[3]:


"""class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        
        input_Q: [batch_size, len_q, d_model] 
        input_K: [batch_size, len_k, d_model]
        input_K: [batch_size, len_v(len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        
        residual, batch_size = input_Q, input_Q.size(0)

        # 1）linear projection [batch_size, seq_len, d_model] ->  [batch_size, n_heads, seq_len, d_k/d_v]
        Q = (self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2))  # Q: [batch_size, n_heads, len_q, d_k]
        K = (self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2))  # K: [batch_size, n_heads, len_k, d_k]
        V = (self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2))  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # 2）mask
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductionAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]

        # 3）concat
        context = torch.cat([context[:, i, :, :] for i in range(context.size(1))], dim=-1)
        
        output = self.concat(context)  # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(d_model).cuda()(output + residual)  # output: [batch_size, len_q, d_model]
    
              
        最后的concat部分，网上的大部分实现都采用的是下面这种方式（也是哈佛NLP团队的写法）
        context = context.transpose(1, 2).reshape(batch_size, -1, d_model)
        output = self.linear(context)
        但是我认为这种方式拼回去会使原来的位置乱序，于是并未采用这种写法，两种写法最终的实验结果是相近的
        
"""


# In[4]:


embeddings = ts.read_h5(h5_file)


# In[5]:


#Train Set 
cluster_rep_unfiltered = ts.read_tsv(train_unfiltered_tsv)
cluster_rep_tolerant = ts.read_tsv(train_tolerant_tsv)
cluster_rep_moderate = ts.read_tsv(train_moderate_tsv)
cluster_rep_strict = ts.read_tsv(train_strict_tsv)


#get ids of validation and test set from fasta
validation_ids = ts.get_ids(validation_fasta)
test_ids = ts.get_ids(test_fasta)

#redundancy reduction (Test and Validation Set) included
train_unfiltered_list, train_unfiltered_IDS, validation_list, test_list = ts.read_json(train_unfiltered_json, validation_ids, test_ids)
train_tolerant_list, train_tolerant_IDS, _, _ = ts.read_json(train_tolerant_json, validation_ids, test_ids)
train_moderate_list, train_moderate_IDS, _, _ = ts.read_json(train_moderate_json, validation_ids, test_ids)
train_strict_list, train_strict_IDS, _, _ = ts.read_json(train_strict_json, validation_ids, test_ids)

train_unfiltered,hn_unfiltered,invalid_shifts_unfiltered,p_u, r_u = ts.get_loaderset(train_unfiltered_list, train_unfiltered_IDS, cluster_rep_unfiltered, embeddings, "Unfiltered", True)
train_tolerant,hn_tolerant,invalid_shifts_tolerant,p_t, r_t = ts.get_loaderset(train_tolerant_list, train_tolerant_IDS, cluster_rep_tolerant, embeddings, "Tolerant", True)
train_moderate,hn_moderate,invalid_shifts_moderate, p_m, r_m = ts.get_loaderset(train_moderate_list, train_moderate_IDS, cluster_rep_moderate, embeddings, "Moderate", True)
train_strict,hn_strict,invalid_shifts_strict,p_s, r_s = ts.get_loaderset(train_strict_list, train_strict_IDS, cluster_rep_strict, embeddings, "Strict", True)


valid,hn_valid,_,_,_ = ts.get_loaderset(validation_list, validation_ids, validation_ids, embeddings, "Validation", True)
test,hn_test,_,_,_ = ts.get_loaderset(test_list, test_ids, test_ids, embeddings, "Test", True)


# In[6]:


#train_unfiltered

# Initialize two empty lists
emb = []
hn = []

# Iterate through the dictionary items
for key, array_3d in train_unfiltered.items():
    # Extract the first and second dimensions from the 3D array
    first_dim_values = [item[1] for item in array_3d]
    second_dim_values = [item[2] for item in array_3d]

    # Append the values to the respective lists
    emb.extend(first_dim_values)
    hn.extend(second_dim_values)


# In[34]:


hn_padd = []
emb_padd = []

longest_list = len(max(hn, key=len))

#padding
for i,sublist in enumerate(hn):
    #hn
    diff = longest_list - len(sublist)
    padd = [[-100, -100]] * diff
    a = sublist + padd
    hn_padd.append(a)
    
    #emb
    d = [[-100] * 1024] * diff
    e = [*emb[i], *d]
    emb_padd.append(e)
    


# In[35]:


emb_1 = torch.tensor(numpy.array(emb_padd)).type(torch.float32)
hn_1 = torch.tensor(numpy.array(hn_padd)).type(torch.float32)

condition_value = torch.tensor([-1.0000e+02, -1.0000e+02])
#indices = torch.all(torch.eq(hn_1, condition_value), dim=2)
indices = torch.all(torch.eq(hn_1, condition_value.view(1, 1, -1)), dim=2).transpose(-2,-1)


# In[58]:


#input_Q=ProtT5_emb, 
#input_K=ProtT5_emb und 
#input_V=H/N-Werte.

#Du kannst die HN Werte entweder direkt einfüttern (2-d Vektor) 
#oder du packst noch ein FNN on-top das dir die HN Werte noch transformiert.

input_Q = emb_1
input_K = emb_1
input_V = hn_1


# In[ ]:


multihead_attn = nn.MultiheadAttention(1024,1,kdim=1024, vdim=2)
attn_output, attn_output_weights = multihead_attn(input_Q, input_K, input_V,need_weights=True, 
                                                  key_padding_mask =indices)


# In[ ]:


max_out = torch.max(attn_output_weights, 1)
max_out_i = max_out.indices


# In[ ]:


import random 

#acc = correct predictions/number of predictions
len_att = len(max_out_i)

comp = (attn_output_weights==max_out_i)
true = (comp == True).sum()
acc = true/len_att
#print(acc)

#random index list to compare accuracy against
res = torch.randperm(len_att)
comp_rndm = (res==max_out_i)
true_rndm = (comp_rndm == True).sum()
acc_rndm = true_rndm/len_att
#print(acc_rndm)

f = open("./out_attention.txt", "w")
f.write("Attention Accuracy (1 head): "+str(acc)+"\nRandom Accuracy: "+str(acc_rndm))
f.close()


# In[ ]:


"""import numpy as np

loss_fn = nn.MSELoss()

target = torch.tensor(np.arange(len(attn_output_weights)))
state = RMSE_loss = torch.sqrt(loss_fn(max_out_i, target))
print(state)"""

