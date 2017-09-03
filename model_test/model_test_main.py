""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:

The entry point for the model_test
"""


import time
from model_test import Model
from loader_all import load_all

#data = load_all("50-10-9.csv")
data = load_all("10-5-4.csv")
model = Model()
start = time.time()
model.train(data=data, iterations=20000, batch_size=30, repeats=2000, precision=0.01)
end = time.time()
print("time :", end - start)
#model.train(data=data, iterations=1500, batch_size=1, repeats=1, precision=0.0001)

#model.train(data=data, iterations=20000, batch_size=1, repeats=1, precision=0.01)
# 6 at 300
#model.train(data=data, iterations=20000, batch_size=1, repeats=1, precision=0.001)
# 5 at 3350 6.1s
#10-5-4
#model.train(data=data, iterations=20000, batch_size=1, repeats=1, precision=0.0001)
# 20 at 2000 70s
#model.train(data=data, iterations=1500, batch_size=30, repeats=500, precision=0.001)
# 5.0 at 2000
#model.train(data=data, iterations=4000, batch_size=30, repeats=1000, precision=0.001)
# 20 at 2400, 5.3s
#model.train(data=data, iterations=10000, batch_size=30, repeats=2000, precision=0.01)
# 6.1 at 1000 (with fluctuations up to 30 after) 1s

#endura 10-5
#model.train(data=data, iterations=1500, batch_size=30, repeats=500, precision=0.0001)
# 6.4 at 1000
