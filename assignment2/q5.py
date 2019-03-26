from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse

from models import RNN, GRU 
from models import make_model as TRANSFORMER
parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')

args = parser.parse_args()

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


########################### 5.2 ###########################################################

def average_Lt(model, valid_data):

    "Getting the data into a matrix where each column represents a batch"
    raw_data = np.array(valid_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // args.batch_size
    data = np.zeros([args.batch_size, batch_len], dtype=np.int32)
    for i in range(args.batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    num_steps = args.seq_len
    epoch_size = (batch_len - 1) // num_steps
    "To have the cross entropy loss for each timestep we cant replace nn.crossentropyloss by these two equivalent functions"
    sm = nn.LogSoftmax()
    nll = nn.NLLLoss()

    loss_moy = torch.zeros(num_steps)
    n = 0
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]

        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda() nn.CrossEntropyLoss(outputs, targets)
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        hidden = repackage_hidden(hidden)

        outputs, hidden = model(inputs, hidden)
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()

        for l in range(num_steps):
            for k in range(args.batch_size):
                o = (outputs[l,k,:]).reshape(1,10000)
                t = (targets[l,k]).reshape(1)
                loss_moy[l] += (nll(sm(o), t)).data.item()
                n+=1
    loss_moy = loss_moy/n

    plt.plot(list(range(1,36)), list(loss_moy), 'go-', label='line 1', linewidth=2)
    plt.title('Average loss at each timestep')
    plt.xlabel('t')
    plt.ylabel('L_t')
    plot = plt
    #plt.savefig('average_time_ste_loss.png')

    return loss_moy, plot, n #the n is to verify if all of the data has been swept through


########################### 5.2 ###########################################################
#Setting arguments
def setattrs(_self, **kwargs):
    for k,v in kwargs.items():
        setattr(_self, k, v)

#The three models
setattrs(args, model= 'RNN', optimizer='ADAM', initial_lr=0.0001, batch_size=20, seq_len=35, hidden_size=1500, num_layers=2, dp_keep_prob=0.35)
setattrs(args, model= 'GRU', optimizer='SGD_LR_SCHEDULE', initial_lr=10, batch_size=20, seq_len=35, hidden_size=1500, num_layers=2, dp_keep_prob=0.35)
setattrs(args, model='TRANSFORMER', optimizer='SGD_LR_SCHEDULE', initial_lr=20, batch_size=128, seq_len=35, hidden_size=512, num_layers=6, dp_keep_prob=0.9)

#To get the hidden layer for the transformer, I added the hidden layer as an output here
class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1), self.transformer_stack(embeddings, mask)

#The function for 5.2
def grad_mean_norm(model, valid_data, vocab_size):

    #Load one mini batch
    raw_data = np.array(valid_data, dtype=np.int32)
    mini_batch = np.zeros([args.batch_size, args.seq_len], dtype=np.int32)
    for i in range(args.batch_size):
        mini_batch[i] = raw_data[0:args.seq_len]
        
    model.eval()


    #extract weight and hidden values of last layer
    grad_moy = torch.zeros(args.hidden_size, vocab_size)
    grad_norm = torch.zeros(args.batch_size)

    inputs = torch.from_numpy(mini_batch.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda() nn.CrossEntropyLoss(outputs, targets)
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    hidden = repackage_hidden(hidden)
    outputs, hidden = model(inputs, hidden)
    
    w_last = model.linears_out[1].weight
    h_last = hidden[1,:,:] 
    n = args.batch_size
    
    for t in range(n):
        h_t = h_last[k,t,:]
        exp_term = torch.exp(torch.mm(w_last,h_t.reshape(args.hidden_size,1)))
        grad_seq =  torch.mm(torch.t(w_last), exp_term) - torch.t(w_last)
        grad_norm[t] = torch.norm(grad_seq, p='fro', dim=None, keepdim=False, out=None).data.item() # L2 norm is default
        grad_moy += grad_seq

    grad_moy = grad_moy / n
    
    #plot grad norm
    itr = [i*35 for i in range(1,21)]
    plt.plot(list(itr), list(grad_norm), 'go-', label='line 1', linewidth=2)
    plt.title('l2 norm of loss gradients at timestep t')
    plt.xlabel('t')
    plt.ylabel('gradL_t')
    plot = plt

    return grad_moy, grad_norm, plot 

#######################################################################################################################################
#################################               5.3                 ###################################################################

args.batch_size = 10

"Getting the data into a matrix where each column represents a batch"   
raw_data = np.array(valid_data, dtype=np.int32)
batch_len = args.seq_len
data = np.zeros([args.batch_size, batch_len], dtype=np.int32)

short_samples = []
long_samples = []

for i in range(args.batch_size):
    n= (len(raw_data)//args.batch_size)*i
    data[i] = raw_data[n : n + batch_len]

    seed = torch.from_numpy(data.astype(np.int64)).transpose(0, 1).contiguous().to(device)
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    hidden = repackage_hidden(hidden)
    outputs, hidden = model(inputs, hidden)

    short_samples[t]= model.generate(seed, hidden, args.seq_len)
    long_samples[t]= model.generate(seed, hidden, 2*args.seq_len)












