import subprocess
import argparse
import sys
import gzip
import cPickle
import pickle
import numpy as np
from config import *
import time
from neural_net import *

class Classifier(object):
    def __init__(self):
        pass

    def train(self):
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference(self):
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)

def create_embeddings():
    data = pickle.load(open('id2word_data.pkl'))
    idx2word = data['idx2word']
    idx2label = data['idx2label']
    num_labels = data['num_labels']   
    # Initialize the word and label embeddings
    word_idx_to_vec = dict((k,2.0*(np.random.random(word_vec_size)-0.5)) for k,_ in idx2word.iteritems())
    label_idx_to_vec = dict((k,2.0*(np.random.random(label_vec_size)-0.5)) for k,_ in idx2label.iteritems())
    return word_idx_to_vec, label_idx_to_vec

def generate_data(raw_data_input, raw_data_output):
    # Used Global Variables: idx2label, label_idx_to_vec, word_idx_to_vec

    num_instances = sum([len(x) for x in raw_data_input])
    data = pickle.load(open('id2word_data.pkl'))
    idx2word = data['idx2word']
    idx2label = data['idx2label']
    num_labels = data['num_labels']
    embeddings = pickle.load(open('embeddings.pkl'))
    word_idx_to_vec = embeddings['word_idx_to_vec']
    label_idx_to_vec = embeddings['label_idx_to_vec']

    # Creating the datasets
    data_input = np.zeros((num_instances, label_vec_size+3*word_vec_size))
    data_output = np.zeros((num_instances, num_labels))
    count = 0
    for word_ids, label_ids in zip(raw_data_input, raw_data_output):
        if len(word_ids) != len(label_ids):
            print('Error: no. of word_ids and label_ids of sentences should be same')
            return
        for idx in range(len(word_ids)):
            if idx == 0:
                prev_label_embed = label_idx_to_vec[start_state_idx]
                prev_word_embed = word_idx_to_vec[start_word_idx]
            else:
                prev_label_embed = label_idx_to_vec[label_ids[idx-1]]
                prev_word_embed = word_idx_to_vec[word_ids[idx-1]]

            if idx == len(word_ids)-1:
                next_word_embed = word_idx_to_vec[end_word_idx]
            else:
                next_word_embed = word_idx_to_vec[word_ids[idx+1]]

            curr_word_embed = word_idx_to_vec[word_ids[idx]]
            
            data_input[count] = np.append(np.append(prev_label_embed, curr_word_embed), np.append(prev_word_embed,next_word_embed))
            data_output[count] = np.eye(num_labels)[label_ids[idx]]

            count = count + 1
    
    # Shuffling the data_input and data_output
    rand_ids = np.random.permutation(num_instances)
    data_input = torch.from_numpy(data_input[rand_ids,:]).type(torch.FloatTensor)
    data_output = torch.from_numpy(data_output[rand_ids,:]).type(torch.FloatTensor)

    # data_input and data_output is a list of numpy arrays.
    return data_input, data_output

def initialize(fileObj):
    pass

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    if train_flag:
        ## Initializing the global variables
        idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
        idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
        num_labels = len(idx2label.keys())
        idx2label[start_state_idx] = 'start'
        idx2label[end_state_idx] = 'end'
        idx2word[start_word_idx] = 'start'
        idx2word[end_word_idx] = 'end'       
        pickle.dump({'idx2word':idx2word, 'idx2label':idx2label, 'num_labels': num_labels},open('id2word_data.pkl','wb'))

        # Initializes word embeddings: 'word_idx_to_vec', 'label_idx_to_vec'
        word_idx_to_vec, label_idx_to_vec = create_embeddings()
        pickle.dump({'word_idx_to_vec':word_idx_to_vec, 'label_idx_to_vec':label_idx_to_vec}, open('embeddings.pkl','wb'))
    else:
        data = pickle.load(open('id2word_data.pkl'))
        idx2word = data['idx2word']
        idx2label = data['idx2label']
        num_labels = data['num_labels']
        embeddings = pickle.load(open('embeddings.pkl'))
        word_idx_to_vec = embeddings['word_idx_to_vec']
        label_idx_to_vec = embeddings['label_idx_to_vec']

    train_input, train_output = generate_data(train_lex, train_y)
    valid_input, valid_output = generate_data(valid_lex, valid_y)
    test_input, test_output = generate_data(test_lex, test_y)

    if train_flag:
        ## Data Preprocessing
        #(-1, batch_size, num_features)
        num_batches = train_input.shape[0] / int(batch_size)
        train_input = train_input[:batch_size*num_batches,:].view(num_batches,batch_size,train_input.shape[1])
        train_output = train_output[:batch_size*num_batches,:].view(num_batches,batch_size,train_output.shape[1])
        mlp = MLP()
        train(mlp, train_input, train_output)
        pickle.dump(mlp, open('trained_model.pkl','wb'))
        return

    mlp = pickle.load(open('trained_model.pkl'))
    # pred_output = torch.max(mlp(Variable(valid_input)).data, 1)[1]
    # true_output = torch.max(valid_output,1)[1]
    # print torch.sum(pred_output == true_output)
    # print torch.sum(pred_output == true_output) / float(pred_output.shape[0])

    # temp = valid_input[0]
    # temp = temp.view(1,-1)
    # print temp.shape
    # print mlp(Variable(temp))
    # print inference(mlp, valid_lex[10])
    # print valid_y[10]

    '''
    To have a look what the original data look like, commnet them before your submission
    '''
    # print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
    # print test_y[0], map(lambda t: idx2label[t], test_y[0])

    '''
    implement you training loop here
    '''

    '''
    how to get f1 score using my functions, you can use it in the validation and training as well
    '''
    predictions_test = [ map(lambda t: idx2label[t], inference(mlp, x)) for x in test_lex ]
    groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

    print test_precision, test_recall, test_f1score



if __name__ == '__main__':
    main()
