import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(facecolor = 'white', edgecolor='white')
ax = fig.add_subplot(111)

xs = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
y1 = [-11123.181578, -12667.149268, -16436.758381, -16527.754344, -18376.067823, -20241.390913, -21932.842926, -22835.232976, -24778.051064, -30761.097369, -35948.111510, -40583.644671]
y2 = [-8302.408524, -9315.726913, -10606.074688, -12034.387262, -13327.083270, -14848.377819, -16004.247611, -17167.484428, -17956.274976, -21066.566457, -23560.563040, -25535.259743]

ax.plot(xs, y1, '-r', label="a = 1/T \nb = 1/T")
ax.plot(xs, y2, '-g', label="a = 50/T \nb = 2/T")

ax.set_xlabel('Number of Topics')
ax.set_ylabel('Perplexity (lower bound)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, prop={'size':12})

plt.savefig('perp2.png')