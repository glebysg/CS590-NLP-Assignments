
word_vec_size = 100
label_vec_size = 20
start_state_idx = 127 # Any random integer > 126
end_state_idx = 128 # Any random integer > 126
start_word_idx = -10
end_word_idx = -11

## Neural Network Configuration
num_features = 3*word_vec_size+label_vec_size
num_hidden_nodes = 200
batch_size = 1000
learning_rate = 0.1
momentum = 0.9
num_epochs = 300
thresh_error = 5*learning_rate * 1e-2

train_flag = False