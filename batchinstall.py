#batchinstall.py
import os
libs = {'torch==1.10.2', 'numpy==1.22.2', 'gym==0.15.7', 'matplotlib==3.5.1'}

try:
    for lib in libs:
        os.system('pip install ' + lib)
    print(lib + 'installed successful.')
except:
    print(lib + 'installed failed.')