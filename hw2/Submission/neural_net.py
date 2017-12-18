# Numpy
import numpy as np

# Pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

## Constants
from config import *

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        data = pickle.load(open('id2word_data.pkl'))
        num_labels = data['num_labels']

        # Note that bias is not added
        self.fc1 = nn.Linear(num_features, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, num_labels)

        # self.fc1 = nn.Linear(num_features, num_labels)

        self.softmax = nn.Softmax()

    def forward(self, data):
        return self.softmax(self.fc2(self.fc1(data)))
        # return self.softmax(self.fc1(data))

def train(net, train_input, train_output):
    criterion = nn.MSELoss() #CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    num_batches = train_input.shape[0]
    running_loss = 100.0  
    # for epoch in range(num_epochs):
    count = 0
    while running_loss > thresh_error:
        count += 1
        running_loss = 0.0
        for batch_idx in range(num_batches):
            inputs = Variable(train_input[batch_idx,:,:])
            labels = Variable(train_output[batch_idx,:,:])
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        running_loss = running_loss / num_batches
        print('Error: ',count,': ', running_loss)
    print('Training Done !!')

def viterbi(prob_list):
    num_states = len(prob_list[0])
    num_words = len(prob_list)
    dptable = torch.zeros(num_states, num_words)
    bp_mat = torch.zeros(dptable.size()).type(torch.LongTensor)
    for idx in range(len(prob_list)):
        if idx == 0:
            dptable[:,idx] = prob_list[idx]
        else:
            temp = torch.max(torch.mul(prob_list[idx], dptable[:,idx-1].contiguous().view(1,-1)), 1)
            dptable[:,idx] = torch.FloatTensor(temp[0]).view(-1,1)
            bp_mat[:,idx] = torch.LongTensor(temp[1]).view(-1,1)
    bp_ids = []
    for idx in range(dptable.shape[1]-1,0,-1):
        if idx == dptable.size()[1]-1:
            bp_ids.append(torch.max(dptable[:,idx], 0)[1][0]) 
        else:
            bp_ids.append(bp_mat[bp_ids[-1],idx+1])
    bp_ids = list(reversed(bp_ids))
    return bp_ids

def inference(net, test_word_ids):
    prob_list = []
    data = pickle.load(open('id2word_data.pkl'))   
    idx2word = data['idx2word']
    idx2label = data['idx2label']
    num_labels = data['num_labels']
    embeddings = pickle.load(open('embeddings.pkl'))
    word_idx_to_vec = embeddings['word_idx_to_vec']
    label_idx_to_vec = embeddings['label_idx_to_vec']

    for word_idx in range(len(test_word_ids)):
        if word_idx == 0:
            prev_label_embed = label_idx_to_vec[start_state_idx]
            prev_word_embed = word_idx_to_vec[start_word_idx]
            next_word_embed = word_idx_to_vec[test_word_ids[word_idx+1]]
            curr_word_embed = word_idx_to_vec[test_word_ids[word_idx]]
            fv = np.append(np.append(prev_label_embed, curr_word_embed), np.append(prev_word_embed,next_word_embed))
            fv = torch.from_numpy(np.reshape(fv,(1,fv.shape[0]))).type(torch.FloatTensor)
            prob_list.append(net(Variable(fv)).data.transpose(1,0))
        else:
            word_prob_mat = torch.zeros(num_labels, num_labels)
            prev_word_embed = word_idx_to_vec[test_word_ids[word_idx-1]]
            curr_word_embed = word_idx_to_vec[test_word_ids[word_idx]]  
            if word_idx == len(test_word_ids)-1:
                next_word_embed = word_idx_to_vec[end_word_idx]
            else: 
                next_word_embed = word_idx_to_vec[test_word_ids[word_idx+1]]          
            count = 0
            for label in idx2label.keys()[:num_labels-len(idx2label.keys())]:
                prev_label_embed = label_idx_to_vec[label]
                fv = np.append(np.append(prev_label_embed, curr_word_embed), np.append(prev_word_embed,next_word_embed))
                fv = torch.from_numpy(np.reshape(fv,(1,fv.shape[0]))).type(torch.FloatTensor)
                word_prob_mat[:,count] = net(Variable(fv)).data.transpose(1,0)
                count += 1
            prob_list.append(word_prob_mat.clone())

    return viterbi(prob_list)

# mlp = MLP()
# print mlp
# params = list(mlp.parameters())
# print len(params)
# print params[0].size(), params[1].size()

# input = Variable(torch.randn(4,1100))
# out = mlp(input)
# print out