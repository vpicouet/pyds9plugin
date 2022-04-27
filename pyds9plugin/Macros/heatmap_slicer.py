import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from mpl_interactions import heatmap_slicer



def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    import numpy as np
    if not operation.lower() in ['sum', 'mean', 'average', 'avg','median']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
        elif operation.lower() in ["median"]:
            ndarray = np.nanmedian(ndarray,-1*(i+1))


    return ndarray

if region is None:
    region=ds9
    region[~np.isfinite(region)] = np.nanmedian(region)

    h, w = region.shape
    n=10
    nrows,ncols=n,n
    max1 = -(h % nrows) if (h % nrows) != 0 else h
    max2 = -(w % ncols) if (w % ncols) != 0 else w
    region = region[:max1, :max2]
    h, w = region.shape
    region = bin_ndarray(region, np.array(np.array(region.shape)/n,dtype=int), operation='median')

region[~np.isfinite(region)] = np.nanmedian(region)

x = np.linspace(0, np.pi, 100)
y = np.linspace(0, 10, 200)
y = np.arange(region.shape[0])
x = np.arange(region.shape[1])

X, Y = np.meshgrid(x, y)
data1 = np.sin(X) + np.exp(np.cos(Y))
data2 = np.cos(X) + np.exp(np.sin(Y))

vmin,vmax=np.array(d.get('scale limits').split(),dtype='float')

data = (region)#,region,region)
fig, axes = heatmap_slicer(
    x,
    y,
    # (data1, data2),
    data,
    slices="both",
    heatmap_names=["Im %i"%(i) for i in range(len(data))],
    labels=("X axis", "Y axis"),
    interaction_type="move",#'click'
    # vmin=vmin,vmax=vmax,
    norm=colors.LogNorm(vmin=vmin, vmax=vmax))
plt.tight_layout()

plt.show()
sys.exit()
