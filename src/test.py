import ObjectDetector
import cv2 as cv

detector = ObjectDetector.Detector()

class printHelper:
  def detectionResultsPrinter(detectionID: float, detectionResults: ObjectDetector.DetectionResults): 
    detectionID += 1
    print('Detections: ', detectionID)
    if not detectionResults.hasTargets:
      print('\t no detection')
    else:
      c = 0
      for i in detectionResults.targets:
        c += 1
      print(f'\tID: {c}, X: {i.x}, Y:{i.y}, YAW:{i.yaw}')

def videoTest():
  video = cv.VideoCapture(0)

  detectionID = 0
  while True:
    _, frame = video.read()
    frame = cv.rotate(frame, cv.ROTATE_180)

    cones = detector.detectCone(frame)
    printHelper.detectionResultsPrinter(detectionID, cones)

    if cv.waitKey(30) == 1:
      break

def staticImageTest():
  image = cv.imread("E:/Projects/cone_detection/78A09405-9EEE-42E1-9AA7-680E614515CD.jpg")

  cones = detector.detectCone(image)
  printHelper.detectionResultsPrinter(0, cones)
  cv.waitKey()

def main():
  staticImageTest()

main()