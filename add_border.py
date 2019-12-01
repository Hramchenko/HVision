import scipy as sp

def add_border(img, color, dx, dy=None):
    if dy is None:
        dy = dx
    shape = list(img.shape)
    shape[1] += dx*2
    shape[0] += dy*2
    result = sp.zeros(shape, dtype=img.dtype)
    result[:, :] = color
    result[dy: img.shape[0] + dy, dx: img.shape[1] + dx] = img
    return result 
