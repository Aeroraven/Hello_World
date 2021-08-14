import pickle

with open(r'C:\Users\huang\Desktop\wen\MRP\MRP\results\ss-test - 副本 (9)\trainlogs.txt', 'rb') as f1:
    d2 = pickle.load(f1)
print(d2)
print(len(d2))