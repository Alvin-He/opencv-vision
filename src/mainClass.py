import numpy as np
import cv2 as cv

HSV_BOUNDS = {
    "CONE_UPPER_BOUND": (30, 255, 255), 
    "CONE_LOWER_BOUND": (17, 100, 150)
}

class DetectionTarget:
  def __init__(self, x: float, y: float) -> None:
    self.x = x
    self.y = y

class DetectionResults: 
  def __init__(self) -> None:
    self.hasTargets = False
    self.targets = []
  
  def append(self, target: DetectionTarget) -> None: 
    self.hasTargets = self.hasTargets or True; 
    self.targets.append(target)

class main:
  def __init__(self):
    #cv.namedWindow("Original Camera", cv.WINDOW_FREERATIO)
    #cv.resizeWindow("Original Camera", (500, 500))
    #cv.namedWindow("Edges", cv.WINDOW_FREERATIO) 
    #cv.resizeWindow("Edges", (500, 500))
    #cv.namedWindow("Final", cv.WINDOW_FREERATIO) 
    #cv.resizeWindow("Final", (500, 500))
    #cv.namedWindow( "HSV Masked", cv.WINDOW_FREERATIO)
    #cv.resizeWindow("HSV Masked", (500, 500))
    pass 
  

  def detectCone(self, frame):
    results = DetectionResults()

    cv.convertScaleAbs(frame, frame, -1, 0.5)
    #cv.imshow("Original Camera", frame)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    blurred = cv.GaussianBlur(hsv, (3,3), 0.8)
    
    coneMasked = cv.inRange(blurred, HSV_BOUNDS["CONE_LOWER_BOUND"], HSV_BOUNDS["CONE_UPPER_BOUND"])
    
    erodeSize = 5
    erodeKernal = cv.getStructuringElement(cv.MORPH_RECT, (erodeSize, erodeSize))
    cv.erode(coneMasked, erodeKernal, coneMasked)
    
    Gx = cv.Sobel(coneMasked, cv.CV_16S, 1, 0)
    Gy = cv.Sobel(coneMasked, cv.CV_16S, 0, 1)
    
    edges = cv.Canny(Gx, Gy, 50, 160)
    
    contours, _= cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contourApproximations = []
    for c in contours: 
      approxCurve = cv.approxPolyDP(c, 10, True)
      contourApproximations.append(approxCurve)
        
    contoursImage = np.zeros_like(edges)
    
    convexHulls = []
    for contour in contourApproximations:
      hull = cv.convexHull(contour)
        
      if 3 > hull.size or hull.size > 13: continue
      boundingBox = cv.boundingRect(contour)
      if (boundingBox[2] * boundingBox[3]) <= 4000: continue
      longerSideLength = boundingBox[2]
      shorterSideLength = boundingBox[3]
      if shorterSideLength > longerSideLength: 
        longerSideLength = boundingBox[3]
        shorterSideLength = boundingBox[2]
      if (longerSideLength / shorterSideLength) > 2: continue

      convexHulls.append(hull)
      results.append(DetectionTarget(boundingBox[0] + boundingBox[2]/2, boundingBox[1] + boundingBox[3]/2))
    
    return results
    #output = frame * cv.cvtColor(coneMasked, cv.COLOR_GRAY2BGR)
    #cv.imshow("HSV Masked", output)
    #cv.drawContours(contoursImage, convexHulls, -1, (255, 255, 255), 2)
    #for h in convexHulls:
        #bbox = cv.boundingRect(h)
        #cv.rectangle(frame,  (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (30, 144, 255), 2)
    #cv.imshow("Final", frame)
    #cv.imshow("Edges", contoursImage)
