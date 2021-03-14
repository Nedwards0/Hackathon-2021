from tkinter import *
import pandas as pd
import torch
from collections import Counter
import numpy as np
from torchtext.data.utils import get_tokenizer
from torch import nn
class TextClassificationModel(nn.Module):
  def __init__(self,num_words,emb_size,num_class):
    super().__init__()
    self.emb_size=emb_size
    self.num_words=num_words
    self.emb1=nn.Embedding(self.num_words,self.emb_size)
    self.lstm1 =nn.LSTM(emb_size,hidden_size=32,batch_first=True)
    self.relu1=nn.ReLU()
    self.lin1 =nn.Linear(32,2)
  def forward(self,batched_data):
    token_embs=self.emb1(batched_data)
    outputs, (h_n,c_n)=self.lstm1(token_embs)
    last_hidden_state=h_n
    last_hidden_state = last_hidden_state.permute(1,0,2)
    last_hidden_state = last_hidden_state.flatten(start_dim=1)

    last_hidden_state = self.relu1(last_hidden_state)

    logits = self.lin1(last_hidden_state)

    return logits
def predict(text):
    text=encode(text)
    text=text.unsqueeze(0)
    output=model(text)
    return output.argmax(1).item()+1

def encode(text):
    tokenizer = get_tokenizer('basic_english')
    vector=[]
    tokens=tokenizer(text)
    for token in tokens:
        if token in res:
            vector.append(hash.get(token))
        else:
            vector.append(hash.get("<unk>"))
    t=torch.tensor(vector)
    return(t)

tokenizer = get_tokenizer('basic_english')

df=pd.read_csv('spam (1).csv')
c=Counter()
#Creating vocabulary of all words over 10 frequency
for x in df["EmailText"]:
  tokens=tokenizer(x)
  tokens
  c.update(tokens)
res = [token for token in c.keys() if c[token] > 10] 
count=0


#encoding into outputs of prefict function
spam_labels = {2: "Spam", 1: "Not Spam"}
hash={}
arr = np.array
count=0
for i in res:
  hash[i]=count
  count=count+1
res.append("<unk>")
hash["<unk>"]=count

#print("Lenght of vocab is : ", len(res))
#print("Lenght of hash is : ", len(hash))
num_class = 2
vocab_size = len(res)
emsize = 64
model = TextClassificationModel(len(res), emsize, num_class)
loaded_params=torch.load("trained_model_weights.pt")
model.load_state_dict(loaded_params)

def calc(*args):
    val1 = v1.get()

    label2["text"] = spam_labels.get(predict(val1))

master = Tk()

label2 = Label(master)
label2.grid(row=4, column=1)




Label(master, text="Main Value").grid(row=0, sticky=E)


v1 = StringVar()


e1 = Entry(master, textvariable=v1)


# Trace when the StringVars are written
v1.trace_add("write", calc)


e1.grid(row=0, column=1)


master.mainloop()


