import numpy as np
import json

f = open("../data/glove.6B.300d.txt","r")
lines = f.readlines()#读取全部内容
embedding={}
for i,(line) in enumerate(lines):
    if i % 5000 ==0:
        break
        print(i)
    line = line.split(' ')
    emb_key = line[0]
    emb_value = np.array(list(map(float,line[1:])))
    embedding[emb_key] = emb_value

# glove embeddings for all the objs.
with open("../data/gcn/objects.txt") as f:
    objects = f.readlines()
    objects_list = [o.strip().lower() for o in objects]

embedding_select={}
for object in objects_list:
    embedding_select[object] = embedding[object]
np.save("../data/glove.6B.300d.npy", embedding_select)
