import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.patches import Polygon

#
from scipy.spatial import Delaunay
import numpy as np

# get arccosine
def arccos_polygon(dx, dy):
    dr = np.sqrt(dx**2 + dy**2)
    if dr == 0:
        return -1.0
    else:
        return np.arccos(dx/dr)

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)

    # HJ: get the points
    points_re = []
    idxs = []
    for idx1, idx2 in edges:
        if idx1 not in idxs:
            idxs.append(idx1)
            points_re.append(points[idx1])
        if idx2 not in idxs:
            idxs.append(idx2)
            points_re.append(points[idx2])
    return edges, np.array(points_re)

#
def sort_by_angle_(pairs):
    if len(pairs) > 0:
        # define start point by minimum y
        start = pairs[np.argmin(pairs[:, 1])]
        # calculate angles according to coordinates
        angles = [arccos_polygon(pair[0] - start[0], pair[1] - start[1]) for pair in pairs]
        # sort by angles
        pairs = pairs[np.argsort(angles)]
    return pairs

#
def get_outline(xs, ys, alpha):
    assert type(xs) == list and type(ys) == list, 'xs or ys type is not list!'
    assert len(xs) == len(ys), 'len(xs) != len(ys)!'

    x_flg = True
    for i in range(len(xs)):
        for j in range(len(xs)):
            if xs[i] != xs[j]:
                x_flg = False
    y_flg = True
    for i in range(len(ys)):
        for j in range(len(ys)):
            if ys[i] != ys[j]:
                y_flg = False

    if x_flg or y_flg:
        return xs, ys

    # coordinates (x, y)
    pairs = np.array([xs, ys]).T
    edges, pairs_re = alpha_shape(pairs, alpha)
    pairs_re = sort_by_angle_(pairs_re)
    # replicate the start point to join the ring
    if len(pairs_re) > 0:
        # pairs_re = np.concatenate((pairs_re, np.array([pairs_re[0]])), axis=0)
        sorted_pairs = pairs_re.T.tolist()
        return sorted_pairs[0], sorted_pairs[1]
    else:
        return xs, ys

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
    # arr = np.array([[0,0],[2,.5],[2.5, 1.25],[2.6,2.8],[1.3,1.1]]).T.tolist()
    # x = arr[0]
    # y = arr[1]

    # x = []
    # y = []
    # for i in range(10):
    #     for j in range(10):
    #         x.append(i)
    #         y.append(j)

    # for i in range(20, 30):
    #     for j in range(20, 30):
    #         x.append(i)
    #         y.append(j)

    x = [-0.25, -0.625, -0.125, -1.25, -1.125, -1.25, 0.875, 1.0, 1.0, 0.5, 1.0, 0.625, -0.2]
    y = [1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 1.875, 1.75, 1.625, 1.5, 1.375, 1.25, 1.25]

    # x = [-2,-4,-6,-5,-8,-5,-8,-9,-3,-0]
    # y = [-3,-6,-2,-9,-5, -3,-7,-6,-9,-8]

    # x = [0.0, 1.0, 2.0, 3.0]
    # y = [0.0, 0.0, 0.0, 1.0]

    # x1, y1 = sort_by_angle(x, y)
    x1, y1 = get_outline(x, y, 1.0)
    xi, yi = interpol_spline(x1, y1, s=0.1, d=1000)
    # xi, yi = interpol(x1, y1)

    fig, ax = plt.subplots()
    ax.plot(xi, yi,'r')
    ax.plot(x1, y1, 'b')
    ax.scatter(x, y, color='k')
    ax.add_patch(Polygon(np.array([xi, yi]).T, closed=True, fill=False, hatch='x', color='r'))
    ax.margins(0.05)
    plt.show()
