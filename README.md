
# Section 1) Practical: To run code


### Pre-requisites:
---------------

1) OpenCV  
2) Tensorflow (version below 1.9)   
3) Module cdc-acm.ko         
4) Cloned models repository of tensorflow    

	git clone	https://github.com/tensorflow/models.git 

*for my commit version used(Jan 2019)*
	
	cd models
	git checkout 21a4ad75c845ffaf9602318ab9c837977d5a9852

### Step 1: Installing dependancies
--------------------------------

	sudo apt install python3-pip python3-dev
	pip3 install --user Cython
	pip3 install --user contextlib2
	pip3 install --user matplotlib
	pip3 install --user pillow
	pip3 install --user lxml    //if this command is throwing error then run > opkg install libxml2-dev libxslt-dev
	pip3 install imutils
		
	sudo apt-get install protobuf-compiler
	
### Step 2: Inside models-master/research/ directory
------------------------------------------------

	cd models/research/
	protoc object_detection/protos/*.proto --python_out=.


### Step 3: Still inside models-master/research/ directory run the following command
---------------------------------------------------------------------------------

	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	
### Step 4: now, connect uARM, start power, connect arm directly to usb port and camera on hub
-----------------------------------------------------------------------------------------------------
	        
	cd models/research/object_detection
copy and replace contents of this repository here

	python3 combined.py (check if usb button is on and camera is connected)

### Step 5

Click on "Register". 

Customer 1: put_name       
Item: Dove,Pears,Medimix (No spaces, and complete names without spelling errors)	

Click Done

Click Exit

Now click "Start"

>wait for some seconds

The objects will be sorted irrespective of sequence. 

After one complete sorting, press EXIT and start again for next, as it will throw error.(to be fixed)

Make sure the timing between each consecutive object is more than 2-3 sec, *after* completetion of arm operation

### For inference on laptop with webcam
	python3 Object_detection_webcam.py


# Section 2) Theory and workflow

Tensorflow object detection API workflow:

Credits: https://github.com/datitran/raccoon_dataset

Overall Summary:
1. Create dataset, ~800*600 preferred 
2. Label using labelimg github that outputs xml file
3. Convert xml to csv using script
4. Convert csv to TF record file using script generateTFrecord.py
5. Create pbtxt file inside training folder and mention all classes in dataset in given format
6. Download selected model configuration(ssd-mobilenet.conf) file and edit the parameters such as various paths, number of classes, augmentations,learning rates etc
7. Copy all these files into object_detection folder in API
8. Run train.py with input model, trainnig directory, dataset and configuration pipeline file with proper arguments
9. Normally should train till 10,000 steps, or till loss < 1 or between 1-2, see progress model training on tensorboard usnig "events.." file inside the training folder  
10. Convert the obtained ckpt file into frozen graph using export_inference_graph.py and giving required arguments

For details and lessons learned, refer [this PPT](https://github.com/shree970/Real-Time-Object-Detection/blob/master/Practical%20Object%20Detection_updated.pdf)

Important commands used:

To install COCO api needed for training


	git clone https://github.com/cocodataset/cocoapi.git
	cd cocoapi/PythonAPI
	make
	cp -r pycocotools ~/object_models/models-master/research/

To start training:

	python object_detection/legacy/train.py --logtostderr --train_dir=object_detection/training/ --pipeline_config_path=object_detection/training/ssdlite.config

After finishing training

	python export_inference_graph.py     --input_type image_tensor     --pipeline_config_path training/ssdlite.config  --trained_checkpoint_prefix training/model.ckpt-29540     --output_directory allfour_inference_graph


To visualise during training

	tensorboard --logdir='training'









	

