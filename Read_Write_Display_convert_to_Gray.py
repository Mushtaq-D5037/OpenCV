import cv2

# load an image using 'imread' specifying the path to image
image = cv2.imread('./images/input.jpg')
# if the image is not loading properly "use full Path"
# C:\\Users\\userName\\Master OpenCV\\images\\digits.png

# display image using 'im.show('Title',imageVariable)
cv2.imshow('FirstImage',image)

# waitkey allows us to input information when an image window is open
# leaving blank it just waits for anykey to be pressed before continuing
# by placing number except(0) we can specify a delay(in milliseconds)
# cv2.waitkey(2)  # waits for 2 milliseconds
cv2.waitkey( )

# closes all open windows
# if not used program hangs
cv2.destroyAllWindows()

# closer look how images are stored 

import numpy as np

print(image.shape) # gives image dimensions (height,width,no.of ColorChannels (RGB)
print('Height of a image:', image.shape[0],'pixels')
print('Width of a image:',  image.shape[1],'pixels')

# alternative to above TWO line of codes
#(height,width) = image.shape[:2]

# save/Write images 
# imwrite('Title',imageVariable)
cv2.imwrite('output_image.jpg',image)
cv2.imwrite('Output.png',image)

# Converting to Grey Scale Images

# why Images are converted to GRAY
# IF we are working with 'CONTOURS ( IMAGE SEGMENTATION TECHNIQUE ),Contours processes only GRAY SCALE IMAGE

# img = cv2.imread('input.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# alternative
img = cv2.imread('input.jpg',0)  # 0 specifies to convert the 'rgb image into grey image

cv2.imshow('GreyImage',img)
cv2.waitkey()
cv2.destroyAllWindows()
