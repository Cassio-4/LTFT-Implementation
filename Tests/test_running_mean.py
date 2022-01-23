import numpy as np
from modules.TrackletManager import Tracklet


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Welford's Method
def stats(x):
    n = 0
    S = 0.0
    m = 0.0
    for x_i in x:
        n = n + 1
        m_prev = m
        m = m + (x_i - m) / n
        S = S + (x_i - m) * (x_i - m_prev)
    return {'mean': m, 'variance': S / n}


t = Tracklet(1, [0, 0, 0, 0], 0.99)

x = x2 = np.linspace(-2, 100, num=6)
x1 = x2 = np.linspace(-1, 0, num=6)
x2 = np.linspace(-1, 1, num=6)

# Normal straightforward mean
l = [x, x1, x2]
for i in range(6):
    sum = x[i] + x1[i] + x2[i]
    print("mean of column {} is {}".format(i, sum/3))

# Using vectorized numpy method in Tracklet class
t.update_mean_enrollable(x)
t.update_mean_enrollable(x1)
t.update_mean_enrollable(x2)
print(t.enrollable_features_running_mean)

