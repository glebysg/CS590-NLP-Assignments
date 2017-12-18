import numpy as np
import argparse
import random
import pickle
from math import *
from string import lower
import operator
import sys


class Classifier(object):
    def __init__(self):
        pass

    def train():
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

class MLP(Classifier):
    # Note: If no. of classes is 2, it is taken as 1.

    def __init__(self, data_input, data_output, hidden_layer_nodes):
        super(MLP, self).__init__()
        # hidden_layer_nodes : [_, _]. List of no. of nodes in hidden layers
        total_input = data_input.copy()
        total_output = data_output.copy()
        train_percent = 0.90
        randperm = np.random.permutation(total_input.shape[1])
        train_ids = randperm[0:int(floor(train_percent*total_input.shape[1]))]
        valid_ids = randperm[int(floor(train_percent*total_input.shape[1])):total_input.shape[1]]

        self.train_input = total_input[:,train_ids]
        self.valid_input = total_input[:,valid_ids]
        self.train_output = total_output[:,train_ids]
        self.valid_output = total_output[:,valid_ids]
        self.config = {}
        # if(total_output.shape[0] == 1): # K: No. of classes
            # self.config['num_classes'] = 2
        # else:
            # self.config['num_classes'] = total_output.shape[0]
        self.config['reg_param'] = 0.03
        self.config['learn_rate'] = 0.01
        self.config['max_iter'] = int(1e5)
        self.config['min_thresh'] = 0.02 * self.config['learn_rate']
        self.config['num_classes'] = total_output.shape[0]
        self.config['num_instances'] = self.train_input.shape[1] # No. of instances
        self.config['num_features'] = self.train_input.shape[0] # data_dimension
        self.config['num_layers'] = len(hidden_layer_nodes) + 1 # No. of layers
        self.config['num_hidden_layers'] = len(hidden_layer_nodes) # No. of hidden layers
        self.config['num_levels'] = len(hidden_layer_nodes) + 2 # No. of levels. [input, hidden_layers, output]
        hidden_layer_nodes.insert(0,self.config['num_features'])
        hidden_layer_nodes.append(self.config['num_classes'])
        self.config['num_nodes'] = hidden_layer_nodes
        # Randomly assigning weights
        self.weights = []
        for idx in range(self.config['num_layers']):
            temp = 2 * (np.random.rand(self.config['num_nodes'][idx+1],self.config['num_nodes'][idx]+1) - 0.5)
            self.weights.append(temp)

        # [self.node_values, self.dnode_values] = self.forward_prop(self.weights, self.train_input)

        # overall_delta = self.back_prop(self.node_values, self.dnode_values, self.weights, self.train_output, self.config['reg_param'])

        # self.print_arch()


    def print_arch(self):
        print('-------- MLP Configuration --------')

        for key in self.config.keys():
            print {key:self.config[key]}
        print {'weights': [weight.shape for weight in self.weights]}
        # print {'node_values': [node.shape for node in self.node_values]}
        # print {'dnode_values': [dnode.shape for dnode in self.dnode_values]}
        print('-----------------------------------')

    def forward_prop(self, weights, data_input):
        # Making a copy
        weights = [weight.copy() for weight in weights]
        data_input = data_input.copy()

        # No self variables change inside the function
        # Weights are not changed.
        node_values = [0 for ww in range(self.config['num_hidden_layers']+2)] # 2 for input and output #
        node_values[0] = data_input
        dnode_values = [0 for ww in range(self.config['num_layers'])] # Adding 1 for output #
        for idx in range(self.config['num_layers']):
            # Appending ones
            temp = np.append(np.ones((1, node_values[idx].shape[1])), node_values[idx], axis=0)
            # Multiplying with weights
            temp = np.dot(weights[idx], temp)
            # Applying Sigmoid
            temp = 1 / (1 + np.exp(-1*temp))
            # Computing gradient of sigmoid
            dtemp = temp * (1 - temp)
            node_values[idx+1] = temp
            dnode_values[idx] = dtemp

        # print [node.shape for node in node_values]
        # print [dnode.shape for dnode in dnode_values]
        return [node_values, dnode_values]

    def back_prop(self, node_values, dnode_values, weights, train_output, reg_param):
        # Making a copy
        node_values = [node.copy() for node in node_values]
        dnode_values = [dnode.copy() for dnode in dnode_values]
        weights = [weight.copy() for weight in weights]
        train_output = train_output.copy()

        # Finding delta - size = num_layers
        delta = [0 for ww in range(self.config['num_layers'])]
        delta[-1] = node_values[-1] - train_output
        for idx in range(len(delta)-2,-1,-1):
            delta[idx] = np.dot(weights[idx+1][:,1:].T, delta[idx+1]) * dnode_values[idx]

        # Finding Delta
        overall_delta = [0 for ww in range(self.config['num_layers'])]
        for idx in range(len(overall_delta)):
            # Appending ones
            temp = np.append(np.ones((1, node_values[idx].shape[1])), node_values[idx], axis=0)
            overall_delta[idx] = np.dot(delta[idx],temp.T) / self.config['num_instances']
            lamb_mat = np.ones(weights[idx].shape)
            lamb_mat[:,0] = 0
            overall_delta[idx] = overall_delta[idx] + reg_param * lamb_mat * weights[idx]
        return overall_delta

    def run(self):
        weights = []
        for idx in range(self.config['num_layers']):
            temp = 2 * (np.random.rand(self.config['num_nodes'][idx+1],self.config['num_nodes'][idx]+1) - 0.5)
            weights.append(temp)

        for iter_idx in range(self.config['max_iter']):
            prev_weights = [weight.copy() for weight in weights]
            [node_values, dnode_values] = self.forward_prop(weights, self.train_input)
            overall_delta = self.back_prop(node_values, dnode_values, weights, self.train_output, self.config['reg_param'])
            for wt_idx in range(len(overall_delta)):
                weights[wt_idx] = weights[wt_idx] - self.config['learn_rate'] * overall_delta[wt_idx]
            # Checking difference
            ssd_weights = [ np.mean(((weights[tidx]-prev_weights[tidx])**2).flatten())**0.5 for tidx in range(len(weights))]
            mean_wt_change = np.mean(np.array(ssd_weights))
            print mean_wt_change
            if(mean_wt_change <= self.config['min_thresh']):
                self.weights = [weight.copy() for weight in weights]
                break
        with open('mlp_weights', 'wb') as fp:
            pickle.dump(self.weights, fp)


    def run_mlp(self):
        lamb_vals = 0.05*np.array(range(1,20))
        res = []
        for idx in range(lamb_vals.size):
            self.config['reg_param'] = lamb_vals[idx]
            self.run()
            output = self.predict(self.valid_input, self.weights)
            # print output.shape, self.valid_output.shape
            res.append(evaluate(output.flatten().tolist(), self.valid_output.flatten().tolist()))
        print res


    def predict(self,data_input, weights):
        [node_values, _] = self.forward_prop(weights, data_input)
        output = node_values[-1]
        output[output<0.5], output[output>=0.5] = int(0), int(1)
        return output

class Perceptron(Classifier):
    def __init__(self, data_input, data_output):
        super(Perceptron, self).__init__()

        total_input = data_input.copy()
        total_input = total_input / (np.linalg.norm(total_input, axis=0)+0.0001)

        total_output = data_output.copy()
        total_output[total_output==0] = -1

        train_percent = 0.90
        randperm = np.random.permutation(total_input.shape[1])
        train_ids = randperm[0:int(floor(train_percent*total_input.shape[1]))]
        valid_ids = randperm[int(floor(train_percent*total_input.shape[1])):total_input.shape[1]]
        self.train_input = total_input[:,train_ids]
        self.valid_input = total_input[:,valid_ids]
        self.train_output = total_output[:,train_ids]
        self.valid_output = total_output[:,valid_ids]
        self.config = {}
        self.config['learn_rate'] = 0.1
        self.config['max_iter'] = int(1e5)
        self.config['min_thresh'] = 0.05*self.config['learn_rate']
        self.config['num_instances'] = self.train_input.shape[1] # No. of instances
        self.config['num_features'] = self.train_input.shape[0] # data_dimension
        # self.weights = 0*(np.random.rand(1,1+self.config['num_features']) - 0.5)
        self.weights = 0*(np.random.rand(1,self.config['num_features']) - 0.5)
        # self.print_arch()


    def print_arch(self):
        print('-------- Perceptron Configuration --------')

        for key in self.config.keys():
            print {key:self.config[key]}
        print {'weights': self.weights.shape}
        print('-----------------------------------')


    def run(self):
        weights = self.weights.copy()
        train_input = self.train_input.copy()
        train_output = self.train_output.copy()
        # train_input = np.append(np.ones((1,self.config['num_instances'])), train_input, axis=0)

        ssd_prev = 100
        count = 0
        for iter_idx in range(self.config['max_iter']):
            prev_weights = weights.copy()
            # Multiplying with weights
            pred_out = np.dot(weights, train_input)
            pred_out[pred_out<0.0], pred_out[pred_out>=0.0] = -1, 1

            temp = train_input[:,(pred_out!=train_output).flatten()]
            temp = temp * train_output[pred_out!=train_output]
            update = np.sum( temp ,axis=1)
            weights = weights + self.config['learn_rate'] * update.T
            ssd = np.mean((prev_weights - weights)**2)**0.5
            print ssd
            if(ssd < ssd_prev):
                count = 0
                ssd_prev = ssd
                fin_weights = weights.copy()
                print fin_weights
            else:
                count = count + 1

            if(ssd < self.config['min_thresh'] or count > 20):
                self.weights = fin_weights.copy()
                break
        with open('perceptron_weights', 'wb') as fp:
            pickle.dump(self.weights, fp)

    def predict(self,data_input, weights):
        data_input = data_input.copy()
        weights = weights.copy()
        # Appending ones
        # temp = np.append(np.ones((1,data_input.shape[1])), data_input, axis=0)
        temp = data_input
        # Multiplying with weights
        temp = np.dot(weights, temp)
        temp[temp<0], temp[temp>=0] = int(0), int(1)
        pred_out = temp
        return pred_out


class ProcessData():
    def __init__(self):
        pass

    @classmethod
    def create_data_splits(cls, out_filename, train_percent):
        if train_percent > 1 and train_percent < 0:
            print('train_percent should be between [0,1] \n')

        with open("sentences.txt") as f:
            data = f.readlines()
        with open("labels.txt") as g:
            labels = [int(label) for label in g.read()[:-1].split("\n")]

        data = [ProcessData.process_line(line) for line in data]

        combined_data = zip(data,labels) # Zipping input and output
        random.shuffle(combined_data) # Shuffling
        total_size = len(combined_data)
        # print total_size

        train_end_idx = int(floor(train_percent*total_size))
        test_end_idx = len(combined_data)

        train_input, train_output = zip(*combined_data[0:train_end_idx])
        test_input, test_output = zip(*combined_data[train_end_idx:test_end_idx])

        total_data = {'train_input':train_input, 'train_output':train_output, 'test_input':test_input, 'test_output':test_output}

        with open(out_filename, 'wb') as fp:
            pickle.dump(total_data, fp)

    @classmethod
    def process_line(cls, line): #Removes all the non alpha numeric charecters in a string
        #########################
        # Removes all the non alpha numeric charecters in the line
        #
        # Input: line - string that you want to process
        #
        # Return: string - line that doesn't contain any non alpha numeric charecters
        #########################
        line = line.replace('\n','')
        line = line.replace('\t',' ')
        words = line.split(' ')
        temp = []
        for word in words:
            word = word.strip()
            word = filter(str.isalnum, word)
            if len(word) == 0:
                continue
            word = lower(word)
            temp.append(word)
        line = ' '.join(temp)
        return line

    @classmethod
    def obtain_feature_words(cls, data, start, end): # Split all the strings to words and find most frequent words
        #########################
        # Split all the strings to words and find most frequent words
        # ... starting from one frequency ('start') to other frequency ('end')
        #
        # Input: data - list of strings
        #        start - start frequency
        #        end - end frequency
        #
        # Return: list of strings/words with most frequency. These words are called feature words.
        #########################
        temp = {}
        data = list(data)
        for line in data:
            st = set()
            words = line.split()
            words = words[2:]
            for word in words:
                word = word.strip()
                if word in st:
                    continue
                elif temp.has_key(word):
                    temp[word] = temp[word] + 1
                else:
                    temp[word] = 1
                st.add(word)
        #temp = sorted(temp, key=temp.__getitem__, reverse = True)
        temp = sorted(temp.items(), key=operator.itemgetter(1), reverse = True)
        temp = temp[start : end+1]
        temp = list(temp)
        feature_words = [temp[i][0] for i in range(len(temp))]
        return feature_words

    @classmethod
    def obtain_feature_vector(cls, line, feature_words): # Return vector of binary values(0,1). Each value corresponds to a word in feature_words .
        #########################
        # Returns a vector of binary values. Each value corresponds to a word in feature_words .
        # 0 - word is absent in 'line', 1 - word is present in the 'line'
        #
        # Input: feature_words - list of strings/words - these are the most frequent words
        #        ... returned by obtain_feature_words()
        #        line - string - can contain non alpha numeric charecters
        #
        # Return: a vector of binary values. Each value in this vector corresponds to a word in 'line'
        #         ... in feature_words.  0 - word is absent in 'line'
        #         ... 1 - word is present in the 'line'
        #########################
        feature_vector = [0 for i in range(len(feature_words))]
        line = ProcessData.process_line(line)
        words = line.split(' ')
        words = words[2:]
        for i in range(len(feature_words)):
            if  words.__contains__(feature_words[i]) :
                feature_vector[i] = 1
            else:
                feature_vector[i] = 0
        return feature_vector

def feature_extractor():
    """
    implement your feature extractor here
    """
    pass

def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    # if(pp!=0):
    #     precision = tp / pp
    # else:
    #     precision = -1
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def main():
    argparser = argparse.ArgumentParser()

    # #Acquiring Data
    out_filename = 'total_data'
    # ProcessData.create_data_splits(out_filename, 0.90)
    with open(out_filename, 'rb') as fp:
        total_data = pickle.load(fp)
    train_input_lines = total_data['train_input']
    train_output = total_data['train_output']
    test_input_lines = total_data['test_input']
    test_output = total_data['test_output']
    feature_words = ProcessData.obtain_feature_words(train_input_lines, 50, 2550)
    train_input = [ProcessData.obtain_feature_vector(line, feature_words) for line in train_input_lines]
    test_input = [ProcessData.obtain_feature_vector(line, feature_words) for line in test_input_lines]
    train_input = np.array(train_input).transpose()
    test_input = np.array(test_input).transpose()
    train_output = np.array([list(train_output)])
    test_output = np.array([list(test_output)])

    ##Training MLP
    # hidden_layer_nodes = [2]
    # mymlp = MLP(train_input, train_output, hidden_layer_nodes)
    # mymlp.run()
    # # Testing MLP
    # # Training Error - MLP
    # train_pred_output = mymlp.predict(train_input, mymlp.weights)
    # print 'train_acc : ', np.mean((train_pred_output.flatten() == train_output.flatten())), np.sum(train_pred_output.flatten()), train_pred_output.flatten().shape
    # print evaluate(train_pred_output.flatten().tolist(), train_output.flatten().tolist())
    # # Test Error - - MLP
    # test_pred_output = mymlp.predict(test_input, mymlp.weights)
    # print 'test_acc : ', np.mean((test_pred_output.flatten()== test_output.flatten())), np.sum(test_pred_output.flatten()), test_pred_output.flatten().shape
    # print evaluate(test_pred_output.flatten().tolist(), test_output.flatten().tolist())

    ##Reading pretrained weights - MLP
    hidden_layer_nodes = [2]
    mymlp = MLP(train_input, train_output, hidden_layer_nodes)
    with open('mlp_weights', 'rb') as fp:
        mymlp.weights = pickle.load(fp)

    ##Training Perceptron
    # myperceptron = Perceptron(train_input, train_output)
    # myperceptron.run()
    # # Training Error - Perceptron
    # train_pred_output = myperceptron.predict(train_input, myperceptron.weights)
    # print 'train_acc : ', np.mean((train_pred_output.flatten() == train_output.flatten())), np.sum(train_pred_output.flatten()), train_pred_output.flatten().shape
    # print evaluate(train_pred_output.flatten().tolist(), train_output.flatten().tolist())
    # # Test Error - Perceptron
    # test_pred_output = myperceptron.predict(test_input, myperceptron.weights)
    # print 'test_acc : ', np.mean((test_pred_output.flatten()== test_output.flatten())), np.sum(test_pred_output.flatten()), test_pred_output.flatten().shape
    # print evaluate(test_pred_output.flatten().tolist(), test_output.flatten().tolist())

    myperceptron = Perceptron(train_input, train_output)
    with open('perceptron_weights', 'rb') as fp:
        myperceptron.weights = pickle.load(fp)

    """
    Testing on unseen testing data in grading
    """
    # argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    # argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")

    parsed_args = argparser.parse_args(sys.argv[1:])
    real_test_sentences = parsed_args.test_data
    real_test_labels = parsed_args.test_labels

    with open(real_test_sentences) as f:
        test_x = f.readlines()
    with open(real_test_labels) as g:
        test_y = [int(label) for label in g.read()[:-1].split("\n")]

    test_x = [ProcessData.process_line(line) for line in test_x]
    test_x = [ProcessData.obtain_feature_vector(line, feature_words) for line in test_x]
    test_x = np.array(test_x).T
    test_y = np.array([list(test_y)])

    predicted_y = mymlp.predict(test_x, mymlp.weights)
    precision, recall, f1 = evaluate(predicted_y.flatten(), test_y.flatten())
    print "MLP results", precision, recall, f1

    predicted_y = myperceptron.predict(test_x, myperceptron.weights)
    precision, recall, f1 = evaluate(predicted_y.flatten(), test_y.flatten())
    print "Perceptron results", precision, recall, f1



if __name__ == '__main__':
    main()
