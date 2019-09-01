import os
import random
import subprocess
import json
import numpy as np
accuracy = []
random_seed = random.randint(10,100000)
final_file_name = "hard_eval.txt"
for i in range(10):
    os.system('python data/division/division_good_distractors.py data/fewrel_val.json 1000 5 1 '+str(i+random_seed) + " division_1K.json")
    x = subprocess.Popen("allennlp evaluate best_model/archive/ division_1K.json --include-package my_library --cuda-device 0 --output-file acc.json", shell=True)

    out, err = x.communicate()
    with open("acc.json") as f:
        data = json.load(f)
        accuracy.append(data['accuracy'])


accuracy = np.array(accuracy)
with open(final_file_name,"w") as f:
    acc = "accuracy = "+ str(np.mean(accuracy))
    std = "std " + str(np.std(accuracy))
    f.write(str(accuracy) +"\n")
    f.write(acc+"\n")
    f.write(std)

print(acc)
print(std)