import numpy as np
data = [[286.0433,
        220.1510,
        302.1160,
        271.4214,
        285.0849]]
label = [0]

sum_loss = 0
for x,l in zip(data,label):
    x = np.array(x)
    x -= np.max(x)
    print(x)
    deno = sum(np.exp(x))
    proba = np.exp(x) / deno
    sum_loss -= np.log(proba[l])

print(sum_loss/len(label))