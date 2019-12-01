import cv2
import scipy as sp

def to_jpeg(img, quality=90):
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
  _, result = cv2.imencode(".jpg", img, encode_param)
  return result

def from_jpeg(data):
  data = sp.frombuffer(data, sp.uint8)
  img = cv2.imdecode(data, -1)
  return img

def save_jpeg(f_name, img, quality=90):
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
  cv2.imwrite(f_name, img, encode_param)
