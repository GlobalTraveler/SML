from numpy import *
from pylab import *

data = fromfile('a012_images.dat', dtype = uint8)
N = 800
t = 28
print(len(data))
data = data.reshape(N, t, t)
fig, ax = subplots()
for i in range(800):
    ax.imshow(data[i,:,:].T)
    pause(1e-3)
