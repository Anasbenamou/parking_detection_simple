import os
import cv2
import pickle 
import numpy as np
import streamlit as st
from numpy import load
from tensorflow import keras
from matplotlib import pyplot
from numpy import expand_dims
from PIL import Image, ImageDraw, ImageFont

def crop_image(img,pts):
  ## (1) Crop the bounding rect
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  croped = img[y:y+h, x:x+w].copy()

  ## (2) make mask
  pts = pts - pts.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

  ## (3) do bit-op
  dst = cv2.bitwise_and(croped, croped, mask=mask)
  return dst

def deploy_model(image_path):
  oc = 0
  no_oc = 0
  cars_img = []
  img = cv2.imread(image_path)
  coordinate = pickle.load(open("train/coordinates.pkl","rb"))
  model      = keras.models.load_model('train/model.h5')
 
  for coor in coordinate:
      pts = np.array(coor, np.int32)
      pts = pts.reshape((-1,1,2))
      #img = cv2.polylines(img,[pts],True,(0,255,0))
      spot = crop_image(img,pts)
      spot = cv2.resize(spot, (45,45), interpolation = cv2.INTER_AREA)
      dst = cv2.cvtColor(spot, cv2.COLOR_BGR2GRAY)
      pred = model.predict(np.asarray([dst]))
      pred = 0 if pred < 0.5 else 1
      if(pred == 0):
        no_oc += 1
        img = cv2.polylines(img,[pts],True,(0,255,0))
      else:
        oc +=  1
        img = cv2.polylines(img,[pts],True,(255,0,0))
        cars_img.append(spot)
  
  return img,oc,no_oc,cars_img


if __name__ == '__main__':
  st.header("Free Parking lot places ")
  st.write("Choose an image:")

  uploaded_file = st.file_uploader("Choose an image...")
	
  if(uploaded_file != None):
    image = Image.open(uploaded_file)	
    cv2.imwrite("img.jpg",np.asarray(image))
    st.image(image, caption='Input Image', use_column_width=True)
    img,oc,no_oc,cars_img = deploy_model("img.jpg")
    st.write("Total spots scanned by this camera is {}".format(no_oc+oc))
    st.write("Number of free spots :{}".format(no_oc))
    st.write("Number of Occupied spots : {}".format(oc))
    st.image(img, caption='Output Image', use_column_width=True) 
    st.write("cars in the parking lot : {}".format(oc))
    with st.container():
      for col in st.columns(1):
          col.image(cars_img,width = 100)
