#!/usr/bin/env python3

# import ntcore
from networktables import NetworkTables
import ObjectDetector
import cv2 as cv

NetworkTables.initialize('10.46.69.2')
mainTable = NetworkTables.getTable("SmartDashboard/VisionServer")
objTable = mainTable.getSubTable("Objects")

def updateNetworkObjDetect(results: ObjectDetector.DetectionResults):
  objTable.putBoolean("HasTargets", results.hasTargets)

  cones = objTable.getSubTable("Cones")
  cubes = objTable.getSubTable("Cubes")

  yawAngles = []
  for i in results.targets:
    yawAngles.append(str(i.yaw))

  yawStringData = ','.join(yawAngles)

  cones.putString("Yaw_Values", yawStringData)

video = cv.VideoCapture(0)

updateNetworkObjDetect(ObjectDetector.DetectionResults())

detector = ObjectDetector.Detector()

while True:
  ret, image = video.read()
  image = cv.rotate(image, cv.ROTATE_180)

  cones = detector.detectCone(image)
  
  updateNetworkObjDetect(cones)

  if cv.waitKey(30) == 1: break