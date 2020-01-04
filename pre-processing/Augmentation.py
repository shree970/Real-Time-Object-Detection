



import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np






Folder_name="augmented_image"
Extension=".jpg"









def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/dove1-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+Extension, image)


def flip_image(image,dir):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name + "/dove1-" + str(dir)+Extension, image)

def invert_image(image,channel):
    # image=cv2.bitwise_not(image)
    image=(channel-image)
    cv2.imwrite(Folder_name + "/dove1-"+str(channel)+Extension, image)


def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/dove1-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dove1-" + str(gamma) + Extension, image)


def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/dove1-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dove1" + str(gamma) + Extension, image)

def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/dove1-" + str(saturation) + Extension, image)


def multiply_image(image,R,G,B):
    image=image*[R,G,B]
    cv2.imwrite(Folder_name+"/dove1-"+str(R)+"*"+str(G)+"*"+str(B)+Extension, image)
def gausian_blur(image,blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Folder_name+"/dove1-"+str(blur)+Extension, image)

def averageing_blur(image,shift):
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(Folder_name + "/dove1-" + str(shift) + Extension, image)
def top_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name + "/dove1-" + "*" + str(shift) + Extension, image)

def edge_image(image,ksize):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Folder_name + "/Edge-"+str(ksize) + Extension, image)
def salt_image(image,p,a):
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Folder_name + "/dove1-"+str(p)+"*"+str(a) + Extension, image)

def paper_image(image,p,a):
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/dove1-" + str(p) + "*" + str(a) + Extension, image)

def salt_and_paper_image(image,p,a):
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1

    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/dove1-" + str(p) + "*" + str(a) + Extension, image)

def contrast_image(image,contrast):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/dove1-" + str(contrast) + Extension, image)
def grayscale_image(image):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/dove1-" + Extension, image)


def scale_image(image,fx,fy):
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(Folder_name+"/dove1-"+str(fx)+str(fy)+Extension, image)

def translation_image(image,x,y):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(x) + str(y) + Extension, image)

def rotate_image(image,deg):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(deg) + Extension, image)

def transformation_image(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(1) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(2) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [30, 175]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(3) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [70, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/dove1-" + str(4) + Extension, image)

def median_blur(image,shift):
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Folder_name + "/dove1-" + str(shift) + Extension, image)



image_file="frame119.jpg"
image=cv2.imread(image_file)




# crop_image(image,0,200,0,255)#(y1,y2,x1,x2)(bottom,top,left,right)
# crop_image(image,50,255,0,255)#(y1,y2,x1,x2)(bottom,top,left,right)
# crop_image(image,0,255,50,255)#(y1,y2,x1,x2)(bottom,top,left,right)
# crop_image(image,0,255,0,200)#(y1,y2,x1,x2)(bottom,top,left,right)
#crop_image(image,100,300,100,350)#(y1,y2,x1,x2)(bottom,top,left,right)
flip_image(image,0)#horizontal
flip_image(image,1)#vertical
flip_image(image,-1)#both

invert_image(image,255)
invert_image(image,200)
invert_image(image,150)
invert_image(image,100)
invert_image(image,50)
add_light(image,1.5)
add_light(image,2.0)
add_light(image,2.5)
add_light(image,3.0)
add_light(image,4.0)
add_light(image,5.0)
add_light(image,0.7)
add_light(image,0.4)
add_light(image,0.3)
add_light(image,0.1)
add_light_color(image,255,1.5)
add_light_color(image,200,2.0)
add_light_color(image,150,2.5)
add_light_color(image,100,3.0)
add_light_color(image,50,4.0)
add_light_color(image,255,0.7)
add_light_color(image,150,0.3)
add_light_color(image,100,0.1)
saturation_image(image,50)
saturation_image(image,100)
saturation_image(image,150)
saturation_image(image,200)

multiply_image(image,0.5,1,1)
multiply_image(image,1,0.5,1)
multiply_image(image,1,1,0.5)
multiply_image(image,0.5,0.5,0.5)

multiply_image(image,0.25,1,1)
multiply_image(image,1,0.25,1)
multiply_image(image,1,1,0.25)
multiply_image(image,0.25,0.25,0.25)

multiply_image(image,1.25,1,1)
multiply_image(image,1,1.25,1)
multiply_image(image,1,1,1.25)
multiply_image(image,1.25,1.25,1.25)

multiply_image(image,1.5,1,1)
multiply_image(image,1,1.5,1)
multiply_image(image,1,1,1.5)
multiply_image(image,1.5,1.5,1.5)


gausian_blur(image,0.25)
gausian_blur(image,0.50)
gausian_blur(image,1)
gausian_blur(image,2)
gausian_blur(image,4)

averageing_blur(image,5)
averageing_blur(image,4)
averageing_blur(image,6)

median_blur(image,3)
median_blur(image,5)
median_blur(image,7)
top_hat_image(image,200)
top_hat_image(image,300)
top_hat_image(image,500)



# scale_image(image,0.3,0.3)
# scale_image(image,0.5,0.5)
# scale_image(image,0.6,0.6)
# scale_image(image,0.8,0.8)
# scale_image(image,0.9,0.9)
# scale_image(image,0.4,0.4)
# scale_image(image,2,2)
# scale_image(image,3,3)
# scale_image(image,4,4)
# scale_image(image,1.5,1.5)
# scale_image(image,2.5,2.5)




translation_image(image,50,50)
translation_image(image,-50,50)
translation_image(image,50,-50)
translation_image(image,-50,-50)

rotate_image(image,90)
rotate_image(image,45)
rotate_image(image,135)
rotate_image(image,180)
rotate_image(image,225)
rotate_image(image,270)
rotate_image(image,330)
rotate_image(image,30)
rotate_image(image,60)
rotate_image(image,100)
rotate_image(image,120)
rotate_image(image,150)
rotate_image(image,190)
rotate_image(image,210)
rotate_image(image,250)
rotate_image(image,290)
rotate_image(image,310)
rotate_image(image,338)


# salt_and_paper_image(image,0.3,0.0006)

#salt_and_paper_image(image,0.5,0.9)

contrast_image(image,15)
contrast_image(image,25)
contrast_image(image,50)
contrast_image(image,30)
contrast_image(image,10)



transformation_image(image)


salt_image(image,0.5,0.009)
salt_image(image,0.5,0.09)
#salt_image(image,0.5,0.9)
salt_image(image,0.5,0.08)
salt_image(image,0.3,0.01)
salt_image(image,0.7,0.04)
salt_image(image,0.5,0.005)
salt_image(image,0.5,0.007)


paper_image(image,0.5,0.009)
paper_image(image,0.2,0.01)
paper_image(image,0.1,0.05)
paper_image(image,0.5,0.007)
paper_image(image,0.6,0.004)
paper_image(image,0.4,0.06)



paper_image(image,0.5,0.09)
#paper_image(image,0.5,0.9)

salt_and_paper_image(image,0.2,0.0009)
salt_and_paper_image(image,0.5,0.0004)
