import torch
import torch.nn as nn

"""
arr=[1,2,3,4,5,6,7,8,9]
arr=arr[:6]
print(arr)
"""

"""
a=[i for i in range(1,25)]
print(a)
b=torch.tensor(a)
c=b.clone().view(2,3,4)
print(b)
print(b.shape)
print(c)
print(c.shape)
print(id(b))
print(id(c))
"""

"""
model=nn.Sequential()
model.add_module('linear1',nn.Linear(3,10))
model.add_module('ReLU1',nn.ReLU())
model.add_module('linear2',nn.Linear(10,100))
model.add_module('ReLU2',nn.ReLU())
model.add_module('Softmax',nn.Softmax())
model.add_module('sequential',nn.Sequential(nn.Linear(100,200),nn.Linear(200,300)))

for i in model:
    print(i)
print("------")
for i in model.children():
    print(i)

# 그래서 children함수쓰는거랑 그냥 model로 for문 도는거랑 무슨차이지?

print("------")

def print_weight(m):
    print(m,type(m))
model.apply(print_weight)

model=model[:4]
print(model)

"""

"""
a=torch.tensor([[1,2],[3,4]])
print(a)
b=a.t()
print(b)
print(id(a))
print(id(b))
gram=torch.mm(a,b)
print(id(gram))
"""

a=[1,2,3,4,5]
print(a[4])
print(a[:4])
