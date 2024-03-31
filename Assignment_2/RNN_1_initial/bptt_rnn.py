"""
# TODO:
    1. 5-fold cross validation
    2. Save the model for each of the folds
    3. Calculate the accuracy, precision, recall, f1-score for each of the folds
"""

import os
import sys
import json
import shutil
import pickle
import itertools
import operator
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
np.random.seed(5)

############################################
############################################
# Define the metrics here



FOLDER_NAME = "history_mine"

try:
    rem = input("Do you want to remove the history_mine folder (y/n) : ")
    if rem == 'y':
        shutil.rmtree('history_mine')
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

try:
    os.makedirs(FOLDER_NAME)
except:
    pass



def precision(result, reference):
    result = np.array(result)
    reference = np.array(reference)
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference):
    result = np.array(result)
    reference = np.array(reference)
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall

def f1_score(p, r): 
    try:
        f1 = (2*p*r)/(p+r)
    except ZeroDivisionError:
        f1 = 0.0
    return f1




############################################

def nparray(lis):
  lis = np.array(lis)
  return lis

def extract_key_values(json_file, key):
    with open(json_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            yield from recursive_extract(data, key)

def recursive_extract(data, key):
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                yield v
            elif isinstance(v, (dict, list)):
                yield from recursive_extract(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from recursive_extract(item, key)

def input_to_5bit_encoder(input_list):
    # Define the mapping from each element to a 5-bit encoder
    mapping = {'0': [1, 0, 0, 0, 0],
               '1': [0, 1, 0, 0, 0],
               '2': [0, 0, 1, 0, 0],
               '3': [0, 0, 0, 1, 0],
               '4': [0, 0, 0, 0, 1]}

    # Convert each sublist into its 5-bit encoder representation
    encoded_list = []
    for sublist in input_list:
        encoded_sublist = [mapping[str(x)] for x in sublist]
        encoded_list.append(encoded_sublist)
    
    return encoded_list

def append_zero_to_sublists(input_list):
    for sublist in input_list:
        sublist.insert(0, 0)

################### Extract the Y values for the Training set
# Example usage:
# json_file_path = r'C:\Users\pande\cs772_repo\Assignment_2\train.jsonl'  # Replace with your JSON file path
json_file_path = './train.jsonl'  # Replace with your JSON file path
key_to_extract = 'chunk_tags'  # Replace with the key you want to extract

values = list(extract_key_values(json_file_path, key_to_extract))

################ Has all the Y values for the training dataset

Y=values
# Here we can print all the values that were extracted, i.e., the first 5 values of the chunk tags
# print(f"Values pertaining to key '{key_to_extract}': {values[:5]}")

##### Do the same for the test dataset
# json_file_path = r'C:\Users\pande\cs772_repo\Assignment_2\test.jsonl'  # Replace with your JSON file path
json_file_path = './test.jsonl'  # Replace with your JSON file path
key_to_extract = 'chunk_tags'  # Replace with the key you want to extract

values = list(extract_key_values(json_file_path, key_to_extract))

################ Has all the Y values for the training dataset

Y_test_full=values
# Here we can print all the values that were extracted, i.e., the first 5 values of the chunk tags
# print(f"Values pertaining to key '{key_to_extract}': {values[:5]}")




############### For the train Array_X values

# Example usage:
# json_file_path =  r'C:\Users\pande\cs772_repo\Assignment_2\train.jsonl'  #'train.jsonl'  Replace with your JSON file path
json_file_path =  './train.jsonl'  #'train.jsonl'  Replace with your JSON file path
key_to_extract = 'pos_tags'  # Replace with the key you want to extract

values = list(extract_key_values(json_file_path, key_to_extract))
X = values
# Here we can print all the values that were extracted, i.e., the first 5 values of the POS tags
# print(f"Values pertaining to key '{key_to_extract}': {values[:5]}")


# Example usage:
input_list_X = X
append_zero_to_sublists(input_list_X) # done to indicate beginning if sentence
# print("Modified List:")
# print(input_list_X)

# Example usage:
input_list = input_list_X
encoded_list = input_to_5bit_encoder(input_list)

# print("Encoded List:", encoded_list[:5])
# quit()


############# The full X dataset for training
# Convert each sublist into a NumPy array -> Has the X for the full dataset
array_X = [np.array(sublist) for sublist in encoded_list]

# print("Encoded List Array:", array_X[:5])
# quit()



############## For the test Array_X test values


# Example usage:
# json_file_path = r'C:\Users\pande\cs772_repo\Assignment_2\test.jsonl'  # Replace with your JSON file path
json_file_path = './test.jsonl'  # Replace with your JSON file path
key_to_extract = 'pos_tags'  # Replace with the key you want to extract

values = list(extract_key_values(json_file_path, key_to_extract))
X = values
# Here we can print all the values that were extracted, i.e., the first 5 values of the POS tags
# print(f"Values pertaining to key '{key_to_extract}': {values[:5]}")


# Example usage:
input_list_X = X
append_zero_to_sublists(input_list_X)
# print("Modified List:")
# print(input_list_X)

# Example usage:
input_list = input_list_X
encoded_list = input_to_5bit_encoder(input_list)

# print("Encoded List:", encoded_list[:5])
# quit()


############# The full X dataset for training
# Convert each sublist into a NumPy array -> Has the X for the full dataset
array_X_test = [np.array(sublist) for sublist in encoded_list]

class RNNNumpy():
    def __init__(self, word_dim, hidden_dim = 1, bptt_truncate = 4):
        # assign instance variable
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # random initiate the parameters - Original version
        #self.I = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)) #WEIGHT FROM CURRENT INPUT TO HIDDEN
        #self.H = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)) #WEIGHT FROM PREVIOUS STATE TO NEXT STATE
        #self.S = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim,word_dim+1)) #WEIGHT FROM PREVIOUS INPUT TO CURRENT HIDDEN
        #self.bias = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(1,1))

        # random initiate the parameters - version - 2
        self.I = np.random.uniform(-1, 1, (hidden_dim, word_dim)) #WEIGHT FROM CURRENT INPUT TO HIDDEN
        self.H = np.random.uniform(-1, 1, (hidden_dim, hidden_dim)) #WEIGHT FROM PREVIOUS STATE TO NEXT STATE
        self.S = np.random.uniform(-1, 1, (hidden_dim,word_dim+1)) #WEIGHT FROM PREVIOUS INPUT TO CURRENT HIDDEN
        self.bias = np.random.uniform(-1, 1,(1,1))

        # bad performing 
        # self.I = np.random.randn(hidden_dim, word_dim) #WEIGHT FROM CURRENT INPUT TO HIDDEN
        # self.H = np.random.randn(hidden_dim, hidden_dim) #WEIGHT FROM PREVIOUS STATE TO NEXT STATE
        # self.S = np.random.randn(hidden_dim,word_dim+1) #WEIGHT FROM PREVIOUS INPUT TO CURRENT HIDDEN
        # self.bias = np.random.randn(1,1)



## 1. forward propagation

def sigmoid(x):
      return np.exp(-np.logaddexp(0, -x))

def forward_propagation(self, x):
    # total num of time steps, len of vector x
    T = len(x)
    # during forward propagation, save all hidden stages in s, S_t = U .dot x_t + W .dot s_{t-1}
    # we also need the initial state of s, which is set to 0
    # each time step is saved in one row in s，each row in s is s[t] which corresponding to an rnn internal loop time
    s = np.zeros((T+1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    x[-1] = np.zeros(self.word_dim+1)
    # output at each time step saved as o, save them for later use
    o = np.zeros((T, 1))
    for t in range(0,T):
        # we are indexing U by x[t]. it is the same as multiplying U with a one-hot vector

        # s[t] = np.tanh(self.I.dot(x[t][1:]) + self.H.dot(s[t-1])+self.S.dot(x[t-1])+self.bias)
        s[t] = np.tanh(self.I.dot(x[t][1:]) + self.H.dot(s[t-1])+self.S.dot(x[t-1])+self.bias)

        o[t] = sigmoid(s[t])
    o = list(itertools.chain(*o))
    s = list(itertools.chain(*s))
    o=o[1:]
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
    o, s = self.forward_propagation(x)
    for i in range(len(o)):
      if o[i]>=0.5:
        o[i] = 1
      else : o[i] = 0
    # print(s)
    return np.array(o)

RNNNumpy.predict = predict
vocabulary_size = 4
# np.random.seed(10)
model = RNNNumpy(vocabulary_size)

'''o, s = model.forward_propagation(nparray(encoded_list[1]))
print(o.shape)
print(o)'''

predictions = model.predict(array_X[0])
# print("True = ",Y[0])
# print("pred = ",predictions.shape)
# print(predictions)


## 2. calculate the loss
'''
the loss is defined as
L(y, o) = -\frac{1}{N} \sum_{n \in N} y_n log(o_n)
'''
def calculate_total_loss(self, x, y):
    L = 0
    # for each sentence ...
    for i in np.arange(len(y)):
        o,_ = self.forward_propagation(x[i])
        o=np.array(o)
        #o = np.array(list(itertools.chain(*o)))
        # we only care about our prediction of the "correct" words
        #correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # add to the loss based on how off we were
        L =  L+(-1 * np.sum(y[i]*np.log(o+1e-6)))
    return L

def calculate_loss(self, x, y):
    # divide the total loss by the number of training examples
    # N = np.sum((len(y_i) for y_i in y))
    N  = np.sum(np.fromiter((len(y_i) for y_i in y), dtype=int))
    return self.calculate_total_loss(x, y)/N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss


#print("Expected Loss for random prediction: %f" % np.log(vocabulary_size))
# print("Actual loss: %f" % model.calculate_loss(array_X[:1], Y[:1]))

## 3. BPTT
'''
1. we nudge the parameters into a direction that reduces the error. the direction is given by the gradient of the loss: \frac{\partial L}{\partial U},
\frac{\partial L}{\partial V}, \frac{\partial L}{\partial W}
2. we also need learning rate: which indicated how big of a step we want to make in each direction
Q: how to optimize SGD using batching, parallelism and adaptive learning rates.

RNN BPTT: because the parameters are shared by all time steps in the network, the gradient at each output depends not only on the calculations of the
current time step, but also the previous time steps.
'''

def bptt(self, x, y):
    T = len(y)
    # perform forward propagation
    o, s = self.forward_propagation(x)
    o = np.array(o)
    # we will accumulate the gradients in these variables
    dLdI = np.zeros(self.I.shape)
    dLdS = np.zeros(self.S.shape)
    dLdH = np.zeros(self.H.shape)
    dLdb = np.zeros(self.bias.shape)
    delta_o = o - y
    dLdb = np.sum(o-y)
    #delta_o[np.arange(len(y)), y] -= 1   # it is y_hat - y
    # for each output backwards ...
    for t in range(0,T):
        delta_t = delta_o[t]    # at time step t, shape is word_dim * hidden_dim
        # initial delta calculation
        #delta_t = self.S.T.dot(delta_o[t]) #* (1 - (s[t] ** 2))
        # backpropagation through time (for at most self.bptt_truncate steps)
        # given time step t, go back from time step t, to t-1, t-2, ...
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print("Backprogation step t=%d bptt step=%d" %(t, bptt_step))
            #print((dLdI[:, x[bptt_step]]+delta_t.T).shape)
            #print(delta_t.T.shape)
            dLdH = dLdH + delta_t*s[bptt_step - 1]
            dLdI[:, x[bptt_step]] = dLdI[:, x[bptt_step]] + delta_t
            # update delta for next step
            delta_t = delta_t*self.H * (1 - s[bptt_step-1]**2)
            dLdS[:, x[bptt_step-1]] = dLdS[:, x[bptt_step-1]] + delta_t
    return [dLdI, dLdS, dLdH,dLdb]

RNNNumpy.bptt = bptt

### 3.1 gradient checking
'''
verify the gradient by its definition:
\frac{\partial{L}}{\partial{\theta}} = \lim_{h \propto 0} \frac{J(\theta + h) - J(\theta - h)}{2h}
'''

def gradient_check(self, x, y, h = 0.001, error_threshold = 0.01):
    # calculate the gradient using backpropagation
    bptt_gradients = self.bptt(x, y)
    # list of all params we want to check
    model_parameters = ["I", "S", "H","bias"]
    # gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # get the actual parameter value from model, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print("performing gradient check for parameter %s with size %d. " %(pname, np.prod(parameter.shape)))
        # iterate over each element of the parameter matrix, e.g. (0,0), (0,1)...
        it = np.nditer(parameter, flags = ['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # save the original value so we can reset it later
            original_value = parameter[ix]
            # estimate the gradient using (f(x+h) - f(x-h))/2h
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # reset parameter to the original value
            parameter[ix] = original_value
            # the gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate the relative error (|x - y|)/(|x|+|y|)
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # if the error is too large fail the gradient check
            if relative_error < error_threshold:
                print("Gradient check error: parameter = %s ix = %s" %(pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for parameter %s passed. " %(pname))

RNNNumpy.gradient_check = gradient_check





## 4. SGD implementation
'''
two step:
1. calculate the gradients and perform the updates for one batch
2. loop through the training set and adjust the learning rate
'''
### 4.1. perform one step of SGD
def numpy_sgd_step(self, x, y, learning_rate):
    dLdI, dLdS, dLdH,dLbias = self.bptt(x, y)

    self.I -= learning_rate * dLdI
    self.S -= learning_rate * dLdS
    self.H -= learning_rate * dLdH[0][0]
    self.bias -= learning_rate * dLbias
RNNNumpy.sgd_step = numpy_sgd_step


### 4.2. outer SGD loop
'''
 - model:
 - X_train:
 - y_train:
 - learning_rate:
 - nepoch:
 - evaluate_loss_after:
'''

def train_with_sgd(model, X_train, y_train, X_test, y_test, learning_rate = 0.005, nepoch = 100, evaluate_loss_after = 5):
    # keep track of the losses so that we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss))
            # adjust the learning rate if loss increases
            
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("setting learning rate to %f" %(learning_rate))
            sys.stdout.flush()
        # for each training example...
        for i in range(len(y_train)):
            # one sgd step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
        # make the inference here
        inference_test(model, X_test, y_test)
    # to return the 2nd index of the losses list
    return model, [tup[1] for tup in losses]


def inference_test(model, X_test, y_test, save_dump = False, fold_number = 0):
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for i, y_true in zip(np.arange(len(y_test)), y_test):
        # print("X_test == ",X_test[i])
        pred = model.predict(X_test[i])
        # print("prediction == ",pred," True == ",y_true)
        accuracy = accuracy_score(y_true, pred)
        prec = precision(y_true, pred)
        rec = recall(y_true, pred)
        f1 = f1_score(prec, rec)
        total_accuracy += accuracy
        total_precision += prec
        total_recall += rec
        total_f1 += f1
    ACCURACY = total_accuracy/len(y_test)
    PRECISION = total_precision/len(y_test)
    RECALL = total_recall/len(y_test)
    F1 = total_f1/len(y_test)
    print("Mean Accuracy == ",ACCURACY)
    print("Mean Precision == ",PRECISION)
    print("Mean Recall == ",RECALL)
    print("Mean F1 == ",F1)

    if save_dump:
        with open(f'history_mine/test_metrics_fold_{fold_number}.txt', 'a') as file:
            file.write(f'Accuracy: {ACCURACY} \n Precision: {PRECISION} \n Recall: {RECALL} \n F1: {F1}\n')


# model = RNNNumpy(vocabulary_size)
# %timeit model.sgd_step(array_X[1], Y[1], 0.005)


print("------------------- Start of Code -------------------")

print("Total number of training samples in array_X == ",len(array_X))
print("Total number of training samples in Y == ",len(Y))

print("Total number of testing samples in array_X_test == ",len(array_X_test))
print("Total number of testing samples in Y_test == ",len(Y_test_full))

print("array_X == ",array_X[1])


'''
FOLD = 5
SAMPLES_PER_FOLD = len(array_X)//FOLD

indices = np.random.permutation(len(array_X))
# print("Indices == ",indices)

array_X = [array_X[i] for i in indices]
Y = [Y[i] for i in indices]

for i in range(FOLD):
    print("------------------------------- FOLD NUMBER = ",i+1)
    print("0 : ",i*SAMPLES_PER_FOLD," add with ",(i+1)*SAMPLES_PER_FOLD," : ",len(array_X))
    X_train = array_X[:i*SAMPLES_PER_FOLD] + array_X[(i+1)*SAMPLES_PER_FOLD:]
    Y_train = Y[:i*SAMPLES_PER_FOLD] + Y[(i+1)*SAMPLES_PER_FOLD:]

    X_test = array_X[i*SAMPLES_PER_FOLD:(i+1)*SAMPLES_PER_FOLD]
    Y_test = Y[i*SAMPLES_PER_FOLD:(i+1)*SAMPLES_PER_FOLD]
    print("Length of X_train == ",len(X_train),"Length of Y_train == ",len(Y_train))
    print("Length of X_test == ",len(X_test),"Length of Y_test == ",len(Y_test))

    # print("X_train == ",X_train,"Y_train == ",Y_train)
    # model, history = train_with_sgd(model, X_train, Y_train, X_test, Y_test, learning_rate = 0.005, nepoch = 10, evaluate_loss_after = 1)

    grad_check_vocab_size = 4
    np.random.seed(42)
    model = RNNNumpy(grad_check_vocab_size, 1, bptt_truncate = 10)

    model, history = train_with_sgd(model, X_train, Y_train, X_test, Y_test, learning_rate = 0.01, nepoch = 100, evaluate_loss_after = 1)

    file_name = f'history_mine/model_fold_{i}.pkl'

    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        print(f'Weights successfully saved to "{file_name}"')

    with open("history_mine/test_logs_fold_{}.txt".format(i), "a") as text_file:
        text_file.write("{} \n".format(history))

    ############# Load the model here and check the inference
    print("Inference from loaded model ==>")
    with open (f'history_mine/model_fold_{i}.pkl', 'rb' ) as f:
        model_loaded = pickle.load(f)
    inference_test(model_loaded, X_test, Y_test, save_dump=True, fold_number=i)


# print("history == ",history)


'''

#### Now check the accuracy, precision, recall, f1-score for the whole dataset
np.random.seed(80)
grad_check_vocab_size = 4
model = RNNNumpy(grad_check_vocab_size, 1, bptt_truncate = 6)
model, history = train_with_sgd(model, array_X, Y, array_X_test, Y_test_full, learning_rate = 1e-5, nepoch = 10, evaluate_loss_after = 1)

file_name = f'history_mine/model_final.pkl'

with open(file_name, 'wb') as file:
    pickle.dump(model, file)
    print(f'Weights successfully saved to "{file_name}"')

############# Load the model here and check the inference
print("Inference from loaded model ==>")

with open (f'history_mine/model_final.pkl', 'rb' ) as f:
    model_final = pickle.load(f)

inference_test(model_final, array_X_test, Y_test_full, save_dump=True, fold_number='final')

