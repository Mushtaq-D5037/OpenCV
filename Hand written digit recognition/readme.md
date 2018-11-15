**Dataset Overview**  
- The Image has 500 samples of each number (0-9)
- total 5000 samples of data
- each individual character has a dimensions:20x20 pixels

**Pipeline**
1. Read Image
2. Convert to Gray Scale
3. Gaussian Blur
4. Canny Edges
5. Extract Contours
6. Get Bounding Rectangle
7. Filter contours
8. Crop Bounding Rectagle of blurred image
9. Binarize and then make into a 20x20 square
10. Flatten to 1x400 array
11. input into classifier


**Convert to GRAY Scale**-- openCV processes only Black&White Images(GRAY IMAGES)

**Blurring:** It is an operation where we average the pixels in a region   
- cv2.blur   
- cv2.GausianBlur  
- cv2.medianBlur   
- cv2.BileteralBlur 
#### Edge Detection algorithm  
- Sobel   
- Laplacian     
- Canny : 
  - Low Error Rate
  - Accurate Detection    
### Contours: Image Segmentation Technique
**Contours are continous lines that covers the full boundary of an object in an image**  
Contours are important in  
- object detection
- shape analysis    
### Finding Contours   
Use a copy of your image e.g. image.copy(), since **findCountours alters** the image   
http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours    

#### mode:   
Contour retrieval mode    
- **CV_RETR_EXTERNAL** retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.   
- **CV_RETR_LIST** retrieves all of the contours without establishing any hierarchical relationships.    
- **CV_RETR_CCOMP** retrieves all of the contours and organizes them into a two-level hierarchy.
  At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. 
  If there is another contour inside a hole of a connected component, it is still put at the top level.   
- **CV_RETR_TREE** retrieves all of the contours and reconstructs a full hierarchy of nested contours. 
  This full hierarchy is built and shown in the OpenCV contours.c demo.

#### method:
contour approximation method (if you use Python see also a note below).

- **CV_CHAIN_APPROX_NONE** stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour 
  will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.   
- **CV_CHAIN_APPROX_SIMPLE** compresses horizontal, vertical, and diagonal segments and leaves only their end points.
  For example, an up-right rectangular contour is encoded with 4 points.    
- **CV_CHAIN_APPROX_TC89_L1, CV_CHAIN_APPROX_TC89_KCOS** applies one of the flavors of the Teh-Chin chain approximation algorithm. 
  See [TehChin89] for details.    
- **offset** – Optional offset by which every contour point is shifted. 
  This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.
