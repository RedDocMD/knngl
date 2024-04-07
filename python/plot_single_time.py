import sys
import matplotlib.pyplot as plt


fname = sys.argv[1]
sizes = []
times = []
with open(fname) as f:
    lines = f.readlines()
    if len(lines) % 2 != 0:
        raise RuntimeError("Invalid number of lines")
    for i in range(len(lines) // 2):
        dline = lines[2 * i][:-1]
        tline = lines[2 * i + 1][:-1]
        size = int(dline.split(' ')[2])
        time = int(tline.split(' ')[2])
        sizes.append(size)
        times.append(time)

plt.stem(sizes, times)
plt.show()
