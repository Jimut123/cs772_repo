import gradio as gr
import numpy as np
import itertools
import pickle

def append_zero_to_sublists(input_list):
    for sublist in input_list:
        sublist.insert(0, 0)


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

def nparray(lis):
    lis = np.array(lis)
    return lis

class RNNNumpy():
    def __init__(self, word_dim, hidden_dim = 1, bptt_truncate = 4):
        # assign instance variable
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # random initiate the parameters
        self.I = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)) #WEIGHT FROM CURRENT INPUT TO HIDDEN
        self.H = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)) #WEIGHT FROM PREVIOUS STATE TO NEXT STATE
        self.S = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim,word_dim+1)) #WEIGHT FROM PREVIOUS INPUT TO CURRENT HIDDEN
        self.bias = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(1,1))

## 1. forward propagation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(self, x):
    # total num of time steps, len of vector x
    T = len(x)
    # during forward propagation, save all hidden stages in s, S_t = U .dot x_t + W .dot s_{t-1}
    # we also need the initial state of s, which is set to 0
    # each time step is saved in one row in s，each row in s is s[t] which corresponding to an rnn internal loop time
    s = np.zeros((T+1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)         # is this needed?
    x[-1] = np.zeros(self.word_dim+1)
    # output at each time step saved as o, save them for later use
    o = np.zeros((T-1, 1))
    for t in range(1,T):
        # we are indexing U by x[t]. it is the same as multiplying U with a one-hot vector
        s[t] = np.tanh(self.I.dot(x[t][1:]) + self.H.dot(s[t-1])+self.S.dot(x[t-1])+self.bias)
        o[t-1] = sigmoid(s[t])
    o = list(itertools.chain(*o))
    s = list(itertools.chain(*s))
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation


def predict(self, x):
    o, s = self.forward_propagation(x)
    print (o)
    for i in range(len(o)):
      if o[i]>=0.5:
        o[i] = 1
      else : o[i] = 0
    # print(s)
    return np.array(o)

RNNNumpy.predict = predict
vocabulary_size = 4
np.random.seed(10)
model = RNNNumpy(vocabulary_size)

# with open (f'model_final.pkl', 'rb' ) as f:
with open (f'rnn_init_5_model_final.pkl', 'rb' ) as f:
    model = pickle.load(f)
    print ('model loaded')



pos_dict = {
            'nn' : 1,
            'dt' : 2, 
            'jj' : 3,
            'ot' : 4
            }


# main ui 
def main_ui(text, pos_tags_text):
    print ('main called')
    pos_tags = pos_tags_text.split(',')

    pos_tags = [pos_dict[x] for x in pos_tags]

    encoded_list = input_to_5bit_encoder(nparray([pos_tags]))
    print (nparray(encoded_list).shape)

    start_ = np.zeros((1, vocabulary_size+1))
    start_[0,0] = 1

    sample = np.concatenate([start_, np.array(encoded_list[0])], axis=0)

    print (sample.shape)

    predictions_num = model.predict(sample)
    print (predictions_num)

    predictions = [str(x) for x in predictions_num.tolist()]

    text_tokens = text.split(' ')

    idx = 0
    new_chunk = []
    all_chunks = []
    while idx < len(predictions_num):
        if predictions_num[idx] == 1:
            new_chunk.append(idx)
            idx += 1
            while idx < len(predictions_num) and predictions_num[idx] ==0:
                new_chunk.append(idx)
                idx +=1

            all_chunks.append(new_chunk)
            new_chunk = []
        else:
            idx += 1   

    result = []

    for chunk in all_chunks:
        # single word chunk and if not noun then skip 
        if len(chunk)==1 and pos_tags[chunk[0]] != 1:
            continue 

        chunk_words = [text_tokens[i] for i in chunk]
        chunk_ = ' '.join(chunk_words)
        result.append(chunk_)

    return ' '.join(predictions), ','.join(result) 
    # return "Hello " + name


inputs = [
    gr.components.Textbox(label="Text", lines=1),
    gr.components.Textbox(label="POS Tags (comma separated)", lines=1)
]

outputs = [
    gr.components.Textbox(label="Predicted Chunk labels"),
    gr.components.Textbox(label="Detected Noun Chunks")
]

title = "Noun Chunk Prediction"

demo = gr.Interface(main_ui, inputs, outputs, title=title, allow_flagging="auto")

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch(share=True)