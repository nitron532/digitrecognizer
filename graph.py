import matplotlib.pyplot as plt
import numpy as np
import time
import re
import math


"""
might wanna add markers on the graph to distinguish between different epochs. 
maybe generate loss graph per epoch? or is that too many graphs? idk
x axis graph gets so large it's almost meaningless, might want to change to rolling window
"""



def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

float_re = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+\.?\d*)([eE][-+]?\d+)?")

def extractFloat(s):
    m = float_re.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
losses = []
batches = []
(line,) = ax.plot([], [], label="Loss", linewidth=1.2)

def updateAxes(xdata,ydata):
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    if len(xdata) > 0:
        ax.set_xlim(0, max(xdata) + 1)  
    WINDOW = 250 #only show last WINDOW batches, maybe calculate amt of batches per epoch and set that as window?
    if len(xdata) > WINDOW:
        xdata = xdata[-WINDOW:]
        ydata = ydata[-WINDOW:]
        ax.set_xlim(min(xdata), max(xdata))
    if len(ydata) > 0:
        yMin = min(ydata)
        yMax = max(ydata)
        if math.isfinite(yMin) and math.isfinite(yMax) and yMax > yMin:
            ax.set_ylim(0.0,(max(1e-8,yMax)* 1.1))
        else:
            ax.relim()
            ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

batchCounter = 0
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(loc="upper right")

logfile = open("cost.txt", "r")
loglines = follow(logfile)
for rawLine in loglines:
    lossValue = extractFloat(rawLine)
    if lossValue is None:
        print(" \"None\" value found!")
        continue
    batchCounter += 1
    batches.append(batchCounter)
    losses.append(lossValue)
    updateAxes(batches,losses)
    losses[-200:]
    plt.pause(0.001)