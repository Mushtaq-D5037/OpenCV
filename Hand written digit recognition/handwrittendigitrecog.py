
import numpy as np
import cv2

img = cv2.imread('C:\\Users\\ISMAIL\\Desktop\\ML_PYthon\\Master OpenCV\\images\\digits.png')
cv2.imshow('Digits Image',img)

# converting to gray (black and white background)
# alternative # img = cv2.imread('images\digits.png',0)        # 0 -- converts the image into gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# lowering the resolution

# Original resolution
# img.shape

# lowered Resolution
small = cv2.pyrDown(img)
# small.shape

cv2.waitKey()
cv2.destroyAllWindows()

# as our image has 50 Rows and 100 columns and each image is a 20x20 pixels
# splitting the image into 50Rows x 100Columns x 20*20 pixels
gray.shape

cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# converting the list data type to numpy array of shape (50,100,20,20)
x = np.array(cells)
x.shape

# train_test_split
# reshaping to flatten into (1 row 400 columns)
train = x[:,:70].reshape(-1,400).astype(np.float32) # Size = (3500,400)
test =  x[:,70:100].reshape(-1,400).astype(np.float32) # Size = (1500,400)

# Create labels for train and test data
k = [0,1,2,3,4,5,6,7,8,9]
# newaxis is used to increase the dimension of the existing array by one more dimension
# 1D array - 2D array
# 2D array - 3D array
train_labels = np.repeat(k,350)[:,np.newaxis] # np.newaxis - converting 1-D array to column vector
test_labels =  np.repeat(k,150)[:,np.newaxis]

# Initailizing KNN
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
retval,result,neighbors,distance = knn.findNearest(test,k=3)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size)
print("Accuracy is = %.2f" % accuracy + "%")

def x_cord_contour(contour):
    """ this function take a contour from findcontours
    it  outputs the x centroid co-ordinates""
    # Formula
    # centroid on Cx = int(M['m10']/M['m00'])
    # centroi on  Cy = int(M['m01']/M['m00'])
    # refer :https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    
    if cv2.contourArea(contour)>10:
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))
    
def makeSquare(not_Square):
    """
    this function takes an image and makes it to square dimensions
    add black pixels as the padding where needed
    """
    black = [0,0,0]
    #img_dim = not_Square.shape
    #height = img_dim[0]
    #width  = img_dim[1]
    (height,width) = not_Square.shape[:2]
    
    if(height == width):
        square = not_Square
        return square
    else:
        doublesize = cv2.resize(not_Square,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
        # as we are doubling the size
        # so multiplying the scaling factor fx along horizontal axis by 2
        # scaling factor fy along vertical axis also by 2
        # cv2.resize(src,fx,fy,inpterpolation)
        # Interpolation -- consturcting new data points with in the range of a discrete set of known data points
        # cv2.INTER_AREA -- good for shrinking or downsampling
        # cv2.INTER_NEARES -- FASTEST
        # cv2.INTER_LINEAR -- Good fro zooming or upsampling (default)
        # cv2.INTER_CUBIC -- BETTER
        # CV2.INTER_LANCZOS4-- BEST
        
        height = height * 2
        width = width * 2
        if (height > width):
            pad = int((height-width)/2)
            doubleSize_Square = cv2.copyMakeBorder(doublesize,0,0,pad,
                                                   pad,cv2.BORDER_CONSTANT,value=black)
        else:
            pad = int((width - height)/2 )
            #print("Padding = ", pad)
            doubleSize_Square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,
                                                   cv2.BORDER_CONSTANT,value=black)
    doubleSize_Square_dim = doubleSize_Square.shape
    print("Sq Height = ", doubleSize_Square_dim[0], "Sq Width = ", doubleSize_Square_dim[1])
    return doubleSize_Square

def resize_to_pixel(dimensions,image):
    """
    re-size an image to the specified dimensions
    """
    buffer_pix = 4
    dimensions = dimensions- buffer_pix
    squared = image
    r = float(dimensions)/squared.shape[1]
    dim = (dimensions,int(squared.shape[0]*r))
    resized = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
    (height_r,width_r) = resized.shape[:2]
    black = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=black)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=black)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=black)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width =  img_dim[1]
    # (height,width) = img_dim.shape[:2]
    
    print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg


# loading new image,preporcessing and classifying the digits
 
# load image
image2 = cv2.imread('C:\\Users\\ISMAIL\\Desktop\\ML_PYthon\\Master OpenCV\\images\\numbers.jpg',0)   # 0 -- converts the image into gray image
# convert it to gray
# gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray2)
cv2.imshow("image", image2)

# Blur Image --Guassian Blur
blur = cv2.GaussianBlur(image2,(5,5),0)
cv2.imshow("blurred", blur)

# finding edges -- canny
edge_2 = cv2.Canny(blur,30,150)
cv2.imshow('edge_canny',edge_2)
cv2.waitkey(0)
cv2.destroyAllWindows()

# Find contours
_,contours,_ = cv2.findContours(edge_2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# sorting contours from left to right
contours = sorted(contours, key = x_cord_contour,reverse =False)

# create empty array to store entire number
full_number = []
# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow("Contours", image)

    if w >= 5 and h >= 25:
        roi = blur[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("final", final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image2, number, (x , y + 155),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image2)
        cv2.waitKey(0) 
        
cv2.destroyAllWindows()
print ("The number is: " + ''.join(full_number))
