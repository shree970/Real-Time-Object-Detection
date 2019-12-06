

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import imutils
import time
import functools

from imutils.video import VideoStream
from imutils.video import FPS

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util



#Initialise uARM swift pro api 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger
logger.setLevel(logger.DEBUG)

swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})


#UI library
from tkinter import *
from tkinter import messagebox


# Name of the directory containing the object detection model we're using
MODEL_NAME = 'allfour_inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labels_new.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `4`, we know that this corresponds to `medimix`.
# Here we use internal utility functions,
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed

video = cv2.VideoCapture(0)
qret = video.set(3,1280)
ret = video.set(4,720)


#wait for camera to initialsie
time.sleep(2.0)



#uARM code for arm action, left and right

def arm_action(cust_no):
    if(cust_no == 1):
        print("Starting Robotic action - Customer 1")
       
        swift.reset(speed=100000, wait=True)
        swift.set_position(x=250, y=0, z=25, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_pump(True)
        swift.set_position(x=250, y=0, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=220, y=70, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=200, y=140, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=180, y=210, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=250, y=210, z=25, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_pump(False)
        swift.flush_cmd()
        swift.set_position(x=250, y=210, z=120, speed=100000,wait = True)
        swift.flush_cmd()
        swift.reset(speed=100000, wait=True)
        swift.flush_cmd()
        time.sleep(0.5)
        
    elif(cust_no ==2):   
        print("Starting Robotic action - Customer 2")
       
        swift.reset(speed=100000, wait=True)
        swift.set_position(x=250, y=0, z=25, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_pump(True)
        swift.set_position(x=250, y=0, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=220, y=-70, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=200, y=-140, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=180, y=-210, z=170, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_position(x=250, y=-210, z=25, speed=100000,wait = True)
        swift.flush_cmd()
        swift.set_pump(False)
        swift.flush_cmd()
        swift.set_position(x=250, y=-210, z=120, speed=100000,wait = True)
        swift.flush_cmd()
        swift.reset(speed=100000, wait=True)
        swift.flush_cmd()
        time.sleep(0.5)
       

    


###################################################################
#HERE ARE ALL SORTING AND GUI UTILITIES

def sort(belt,list1):
    out = []
    for item in belt:
        if item in list1[0]:
            out.append(1)
            list1[0].remove(item)
        elif item in list1[1]:
            out.append(2)
            list1[1].remove(item)
        #elif item in list1[2]:
        #    out.append(3)
        #    list1[2].remove(item)
        #elif item in list1[3]:
        #    out.append(4)
        #    list1[3].remove(item)
        else:
            out.append('None');
    
    return out[0]
#def callback(event):
        #web

class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master=master
        pad=3
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)            
    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom


#Creating Form-------------------------------------------------------------------------------------------------------------
field1 = 'Customer 1', 'Items','Customer 2', 'Items'
cust_list = []

def fetch(entries):
    #print(entries)
   temp = []
   temp_final = []
   
   for entry in entries:
        temp.append(entry[1].get())
  
       #field = entry[0]
      #text  = entry[1].get()
   #if(cust1[1]):
   temp_final.append(temp[1].split(','))
   temp_final.append(temp[3].split(','))
   #temp_final.append(temp[5].split(','))
   #temp_final.append(temp[7].split(','))
   
  # if(cust2[0]):
        #cust2_final = cust2[1].split(',')
   
   #print(temp_final)
   #print(sort(belt,temp_final))
   
   #cust_list = temp_final.copy()
   cust_list.append(temp_final)
  # return cust_list

   root.destroy
   return temp_final
   
def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=20, text=field, anchor='w',font=('Helvetica', '50'))
      ent = Entry(row,font=('Helvetica', '25'))
      row.pack(side=TOP, fill=X, padx=5, pady=15)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
    
   return entries

def register_page():
   root1 = Tk()
 
   app=FullScreenApp(root1)
   ents = makeform(root1, field1)
   root1.bind('<Return>', (lambda event, e=ents: fetch(e)))   
   b1 = Button(root1, text='Done',command=(lambda e=ents: fetch(e)), bd=1,bg = "green",height = 3, width=12,font=('Helvetica', '25') ).place(x=50,y=400)
  
   b2=Button(root1, text="Exit", bd=1,bg = "green", command = root1.destroy,height = 3, width=12,font=('Helvetica', '25') ).place(x=300,y=400)
   root1.mainloop()
  
#Home Page
#----------------------------------------------------------------------------------------------------------------
def close_window(): 
    root.destroy()


#When clicked on start    
def start_command(): 
    cust_list_final=cust_list[0]
    print(cust_list_final)  
    objects1 = {}
    a=[]
    fps = FPS().start()    
    countnone=0
    countobj=0  
#-----------    
    while(True):
        #UI initialize


        #################

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)
        
        #IMPORTANT STEP
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        #get scores
            
        objects = []
        
        #setting threshold value
        threshold= 0.80
        

        #looping in values obtained in classes 

        for index, value in enumerate(classes[0]):
          object_dict = {}
        


          #If first element of scorep[] is higher than threshold,that means an object is present 
          if scores[0, index] > threshold:
            object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                scores[0, index]
            objects.append(object_dict)
            

            countobj=countobj+1 #increase obj flag
            countnone=0 #reset none flag


            #Map dictionary value and get the name of object
            if(object_dict.keys() != objects1.keys())  and countobj==10: #If object is consistent for 10 frames the  only go forward          
             print(objects)

             for key in object_dict:
                if key in [b'pears']:
                    soap=['Pears']
                    arm_action(sort(soap,cust_list_final))
                    objects1 = object_dict.copy()
                elif key in [b'dove']:
                    soap=['Dove']                    
                    arm_action(sort(soap,cust_list_final))
                    objects1 = object_dict.copy()
                elif key in [b'moti']:
                    soap=['Moti']
                    arm_action(sort(soap,cust_list_final))
                    objects1 = object_dict.copy()
                elif key in [b'medimix']:
                    soap=['Medimix']
                    arm_action(sort(soap,cust_list_final))
                    objects1 = object_dict.copy()
				            
            #If prediction scores are less tha  0.5, the  there is certainly no object present
          elif scores[0, 0]<0.5:
            
            countnone=countnone+1 #increase none flag
            
            if countnone==1000:   #if none flag is up for more than 1000 frames(1-2 secs), reset obj flag for new object to come
                countobj=0        #this prevents not detection of two consecutive same objects
                objects1={} 





        # Draw the results of the detection as bounding box (aka 'visulaize the results')
        frame=vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)

        
     



        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        fps.update() 
       








    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Clean up
    video.release()
    cv2.destroyAllWindows()
     




####################################################################
if __name__ == '__main__':
    #GUI STUFF, INITIAL DISPLAY SCREEN
    root = Tk()
    app=FullScreenApp(root)
    cwgt=Canvas(root)
    cwgt.pack(expand=True, fill=BOTH)
    title=Button(cwgt, text="AI Powered Robotic Arm", bd=1,bg = "yellow",height = 6, width=210,font=('Helvetica', '40')).place(x=0,y=0)
    #title.bind("<Button-1>",callback)
    image1=PhotoImage(file="final.png")
    cwgt.img=image1
    cwgt.create_image(5, 50, anchor=NW, image=image1)
    b1=Button(cwgt, text="Exit", bd=1,bg = "red", command = close_window,height = 3, width=10,font=('Helvetica', '15') ).place(x=1150,y=25)
    #cwgt.create_window(15,15, window=b1, anchor=NW)
    b2=Button(cwgt, text="Start", bd=1,bg = "green", command = start_command,height = 3, width=12,font=('Helvetica', '25') ).place(x=300,y=300)
    b3=Button(cwgt, text="Register", bd=1,bg = "Blue", command = register_page,height = 3, width=12,font=('Helvetica', '25') ).place(x=750,y=300)
    root.mainloop()












