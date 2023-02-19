import numpy as np
import cv2 as cv

import ObjDetectHelpers as Helpers

HSV_BOUNDS = {
    "CONE_UPPER_BOUND": (30, 255, 255),
    "CONE_LOWER_BOUND": (17, 100, 150)
}


class DetectionTarget:
  def __init__(self, yawToTarget: float) -> None:
    self.type = '' #unimplemented 
    self.yaw = yawToTarget


class DetectionResults:
  def __init__(self) -> None:
    self.hasTargets = False
    self.targets: list[DetectionResults] = []

  def append(self, target: DetectionTarget) -> None:
    self.hasTargets = self.hasTargets or True
    self.targets.append(target)


radToDegreesConversionFactor = 180 / np.pi
# calculates the # of degrees to turn to align a coordinate to centerline, assuming frame is upright 
# positive is left, negative is right
def calculateYaw(objX: float, objY: float, frameWidth: float, frameHeight:float) -> float: 
  # assuming the frame is not flipped/tilted aka, frame in up right 
  opposite = (frameWidth / 2) - objX
  adjacent = frameHeight - objY

  return np.arctan(opposite / adjacent) * radToDegreesConversionFactor

class Detector:
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

    blurred = cv.GaussianBlur(hsv, (3, 3), 0.8)

    coneMasked = cv.inRange(
        blurred, HSV_BOUNDS["CONE_LOWER_BOUND"], HSV_BOUNDS["CONE_UPPER_BOUND"])

    erodeSize = 5
    erodeKernal = cv.getStructuringElement(
        cv.MORPH_RECT, (erodeSize, erodeSize))
    cv.erode(coneMasked, erodeKernal, coneMasked)

    Gx = cv.Sobel(coneMasked, cv.CV_16S, 1, 0)
    Gy = cv.Sobel(coneMasked, cv.CV_16S, 0, 1)

    edges = cv.Canny(Gx, Gy, 50, 160)

    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contourApproximations = []
    for c in contours:
      approxCurve = cv.approxPolyDP(c, 10, True)
      contourApproximations.append(approxCurve)

    contoursImage = np.zeros_like(edges)

    convexHulls = []
    for contour in contourApproximations:
      hull = cv.convexHull(contour)

      if 3 > hull.size or hull.size > 13:
        continue
      boundingBox = Helpers.BoundingRect(cv.boundingRect(contour))
      if (boundingBox.area()) <= 4000: 
        continue
      longerSideLength = boundingBox.width
      shorterSideLength = boundingBox.height
      if shorterSideLength > longerSideLength:
        longerSideLength = boundingBox.height
        shorterSideLength = boundingBox.width
      if (longerSideLength / shorterSideLength) > 2:
        continue
      

      
      convexHulls.append(hull)
      cords = boundingBox.center()
      frameHeight, frameWidth, _ = frame.shape
      yawToTarget = calculateYaw(*cords, frameWidth, frameHeight)
      results.append(DetectionTarget(yawToTarget))

    return results
    #output = frame * cv.cvtColor(coneMasked, cv.COLOR_GRAY2BGR)
    #cv.imshow("HSV Masked", output)
    #cv.drawContours(contoursImage, convexHulls, -1, (255, 255, 255), 2)
    #for h in convexHulls:
    #bbox = cv.boundingRect(h)
    #cv.rectangle(frame,  (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (30, 144, 255), 2)
    #cv.imshow("Final", frame)
    #cv.imshow("Edges", contoursImage)
