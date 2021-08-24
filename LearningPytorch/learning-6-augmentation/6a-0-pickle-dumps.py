import pickle
r=None
with open('exp-2-valid.txt', 'rb') as f:
    r=pickle.load(f)
print(r)