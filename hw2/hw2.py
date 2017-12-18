import subprocess
import argparse
import sys
import gzip
import cPickle

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

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data

    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)


    for idx in range(len(train_set[0])):
        print len(train_set[0][idx]), len(train_set[1][idx]), len(train_set[2][idx])
    print train_set[0][0]
    print train_set[1][1]
    print train_set[2][2]

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    # print sorted(dicts['labels2idx'].values())

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())

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
    # predictions_test = [ map(lambda t: idx2label[t], myrnn.inference(x)) for x in test_lex ]
    # groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    # words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    # test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

    # print test_precision, test_recall, test_f1score



if __name__ == '__main__':
    main()
