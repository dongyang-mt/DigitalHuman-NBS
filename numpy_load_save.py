import os
import numpy as np

filepath = "./eval_constant/sequences/greeting.npy"
folderseq = "./eval_constant/sequences/"
np_array = np.load(filepath)
print(np_array.shape)

for filename in os.listdir("./eval_constant/sequences/"):
    base, ext = os.path.splitext(filename)
    if ext not in [".npy", ".npz"]:
        continue
    filepath = os.path.join(folderseq, filename)
    np_array = np.load(filepath)
    print(filename, "shape:", np_array.shape)

def load_print_numpy_info(filepath):
    np_array = np.load(filepath)
    print(filepath, np_array.shape)

# demo_greeting_smpl/weight.npy (6890, 24)
# blender_scripts/tableau_color.npy (20, 3)
load_print_numpy_info("demo_greeting_smpl/weight.npy")
load_print_numpy_info("blender_scripts/tableau_color.npy")
load_print_numpy_info("eval_constant/sequences/dance.npy")


x = [1, 2, 3, 4, 5]
i = 0
i = x[i] = 3 # means: i=3; x[i]=3
print(i)
print(x)
