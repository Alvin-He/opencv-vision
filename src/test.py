import mainClass as ObjectDetector
import cv2 as cv

detector = ObjectDetector.main()

video = cv.VideoCapture(0); 

detectionID = 0
while True:
    _, frame = video.read()
    frame = cv.rotate(frame, cv.ROTATE_180)

    cones = detector.detectCone(frame)
    detectionID += 1
    print('Detections: ', detectionID)
    if not cones.hasTargets: print('\t no detection')
    else:
        c = 0
        for i in cones.targets:
            c += 1
            print('\t ID: %s    X: %s, Y %s' % (c, i.x, i.y))

    if cv.waitKey(30) == 1: break
    
