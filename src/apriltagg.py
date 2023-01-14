import cv2 as cv

import apriltag

cap = cv.VideoCapture(0); 

cv.namedWindow('Image')

options = apriltag.DetectorOptions('tag16h5')
detector = apriltag.Detector(options); 

while True:
    ret, processed = cap.read(); 

    processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY); 

    results = detector.detect(processed)
    pc = 0 
    for r in results:
        print(r.tag_id)
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv.line(processed, ptA, ptB, (0, 255, 0), 2)
        cv.line(processed, ptB, ptC, (0, 255, 0), 2)
        cv.line(processed, ptC, ptD, (0, 255, 0), 2)
        cv.line(processed, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv.circle(processed, (cX, cY), 5, (0, 0, 255), -1)

    cv.imshow('Image', processed)

    if cv.waitKey(30) == 1: 
        break 