import matplotlib.pyplot as plt
import numpy as np
import time

def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

logfile = open("cost.txt", "r")
loglines = follow(logfile)
for line in loglines:
    print(line)


# fig, ax = plt.subplots()
# ax.plot([1,2,3,4],[1,4,2,3])
# plt.show()