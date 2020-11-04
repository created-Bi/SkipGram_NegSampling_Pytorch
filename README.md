# SkipGram_NegSampling_Pytorch
A simple example using skip_gram to obtain the representation of words.
In order to simplify the process of computing the loss, negative sampling(NegS) will be used.

## Requirements
Pytorch >= 1.3.0
tqdm
sklearn

## 😀Get Started
Get this code. git clone https://github.com/created-Bi/SkipGram_NegSampling_Pytorch.git

### 😊1. Corpus
  In this case, we use the PTB data from torchnlp.
  You can see more details from https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html
### 😚2. How to implement ?
#### 2.1 Skip_Gram
  Skip Gram use the center word to predict the context words.
#### 2.2 Negative Sampling
  During computing the loss, we just need a few of the random words, which are not near the center word.

## Intro to the main classes
### 1. SkipGramModel: 
 The main design of the skip_gram algorithm
### 2. create_sample_table: 
  Because of the NegS, we need to create a table according to the frequency of the words occuring in the corpus.
### 3. get_ptb_data: 
  We train the model with the most common corpus named PTB, and <penn_treebank_dataset(train=True)> represents you will download the train data.
### 4. get_center_context_pairs:
  When we train the model, we need a (or a batch of) <center_word, context_word> pair. 
