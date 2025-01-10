from numpy import load

data = load('/home/aayusphere/Embeddings/poj/milepost/trainO0/1/1-14.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])