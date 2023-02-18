#!/usr/bin/env python3

# import ntcore
from networktables import NetworkTables
import mainClass as ObjectDetector
import cv2 as cv

def updateNetworkObjDetect(results: ObjectDetector.DetectionResults):
  objTable.putBoolean("HasTargets", results.hasTargets)

  cones = objTable.getSubTable("Cones")
  cubes = objTable.getSubTable("Cubes")

  xPoses = []
  yPoses = []
  for i in results.targets:
    xPoses.append(str(i.x))
    yPoses.append(str(i.y))

  xStringData = ','.join(xPoses)
  yStringData = ','.join(yPoses)

  cones.putString("X_Values", xStringData)
  cones.putString("Y_Values", yStringData)


NetworkTables.initialize('10.46.69.2')
mainTable = NetworkTables.getTable("SmartDashboard/VisionServer")
objTable = mainTable.getSubTable("Objects")

video = cv.VideoCapture(0)

updateNetworkObjDetect(ObjectDetector.DetectionResults())

detector = ObjectDetector.main()

while True:
  ret, image = video.read()
  image = cv.rotate(image, cv.ROTATE_180)

  cones = detector.detectCone(image)
  
  updateNetworkObjDetect(cones)

  if cv.waitKey(30) == 1: break