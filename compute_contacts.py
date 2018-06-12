import numpy as np
import sys

conf_file = sys.argv[1]
label_pos = int(sys.argv[2])

conf = np.array(eval(open(conf_file).read()))

for i in range(len(conf)):
    d = np.linalg.norm(conf[i] - conf[label_pos])
    if d < 1.01:
        print label_pos, i
