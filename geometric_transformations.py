import scipy as sp
import cv2

def toHomogenious(X):
  X = sp.hstack((X, sp.ones((X.shape[0], 1))))
  return X
  
def toEuclidean(X):
  W = X[:,2]
  X = X/W[:, sp.newaxis]
  return X[:, 0:2]

def points2DTransform(points, M):
    points = toHomogenious(points)
    points = M.dot(points.T).T
    points = toEuclidean(points)
    return points

def identityMatrix():
  M = sp.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
                ])
  return M  

def xShearMatrix(image_height, angle):
  lambda_ = sp.tan(angle/180.0*sp.pi)
  delta_width = -lambda_*image_height
  dx = 0
  if lambda_ < 0:
    dx = delta_width;
  M = sp.array([[1, lambda_, dx], [0, 1, 0], [0, 0 ,1]])
  return M 

def xFlipMatrix(image_width):
  M = sp.array([
                [-1, 0, image_width],
                [0, 1, 0],
                [0, 0, 1]
                ])
  return M  

def yFlipMatrix(image_height):
  M = sp.array([
                [1, 0, 0],
                [0, -1, image_height],
                [0, 0, 1]
                ])
  return M  

def rotationMatrix(angle):
  angle = sp.deg2rad(angle)
  M = sp.array([
                [sp.cos(angle), -sp.sin(angle), 0],
                [sp.sin(angle),  sp.cos(angle), 0],
                [0,              0,             1]
                ])
  return M

def translationMatrix(x, y):
  M = sp.array([
                [1, 0, x],
                [0, 1, y],
                [0, 0, 1]
                ])
  return M

def scaleMatrix(x, y):
  M = sp.array([
                [x, 0, 0],
                [0, y, 0],
                [0, 0, 1]
                ])
  return M

def transformedImageBoundaries(w, h, M):
  """minx, miny, maxx, maxy"""
  pts = sp.array([[0, 0], [w, 0], [0, h], [w, h]])
  pts = points2DTransform(pts, M)
  r = sp.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
  return r

def calcFittingParameters(img, M):
  h = img.shape[0]
  w = img.shape[1]
  B = transformedImageBoundaries(w, h, M)
  T = translationMatrix(-B[0], -B[1])
  M = T.dot(M)
  B = sp.int32(B)
  newSize = (B[2] - B[0], B[3] - B[1])
  return sp.float32(M), newSize




