import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.patches import Polygon

def arccos_polygon(dx, dy):
    dr = np.sqrt(dx**2 + dy**2)
    if dr == 0:
        return -1.0
    else:
        return np.arccos(dx/dr)

# sort coordinates according to angles
def sort_by_angle(xs, ys):
    if len(xs) == len(ys) and type(xs) == list and type(ys) == list:
        # coordinates (x, y)
        pairs = np.array([xs, ys]).T
        # define start point by minimum y
        start = pairs[np.argmin(pairs[:, 1])]
        # calculate angles according to coordinates
        angles = [arccos_polygon(pair[0] - start[0], pair[1] - start[1]) for pair in pairs]
        # sort by angles
        pairs = pairs[np.argsort(angles)]
        # replicate the start point to join the ring
        pairs = np.concatenate((pairs, np.array([pairs[0]])), axis=0)
        sorted_pairs = pairs.T.tolist()
        return sorted_pairs[0], sorted_pairs[1]
    else:
        return [], []

def interpol_spline(x, y, s=0.1, d=100):
    #create spline function
    f, u = interpolate.splprep([x, y], s=s, per=True)
    #create interpolated lists of points
    xi, yi = interpolate.splev(np.linspace(0, 1, d), f)
    return xi, yi

def interpol(x, y, d=10):
    t = np.arange(len(x))
    ti = np.linspace(0, t.max(), d * t.size)
    xi = interpolate.interp1d(t, x, kind='cubic')(ti)
    yi = interpolate.interp1d(t, y, kind='cubic')(ti)
    return xi, yi

if __name__ == "__main__":
    arr = np.array([[0,0],[2,.5],[2.5, 1.25],[2.6,2.8],[1.3,1.1]]).T.tolist()
    x = arr[0]
    y = arr[1]

    x = [-0.25, -0.625, -0.125, -1.25, -1.125, -1.25, 0.875, 1.0, 1.0, 0.5, 1.0, 0.625, -0.2]
    y = [1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 1.875, 1.75, 1.625, 1.5, 1.375, 1.25, 1.25]

    # x = [-2,-4,-6,-5,-8,-5,-8,-9,-3,-0]
    # y = [-3,-6,-2,-9,-5, -3,-7,-6,-9,-8]

    x1, y1 = sort_by_angle(x, y)
    xi, yi = interpol_spline(x1, y1, s=0.1, d=1000)
    # xi, yi = interpol(x1, y1)

    fig, ax = plt.subplots()
    ax.plot(xi, yi,'r')
    ax.plot(x1, y1, 'b')
    ax.add_patch(Polygon(np.array([xi, yi]).T, closed=True, fill=False, hatch='x', color='r'))
    ax.margins(0.05)
    plt.show()
