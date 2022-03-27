import cv2
import numpy as np

MIN_MATCHES = 125

orb = cv2.ORB_create()
bruteForceMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
video_capture = cv2.VideoCapture(0)

referenceImage = cv2.imread('reference.jpg', 0)
referenceImage = cv2.resize(referenceImage, (400, 400))
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)

while True:
    _, frame = video_capture.read()
    frame = cv2.resize(frame, (400, 400))
    framePts, frameDsc = orb.detectAndCompute(frame, None)

    matches = bruteForceMatcher.match(referenceImageDsc, frameDsc)
    matches = sorted(matches, key=lambda x: x.distance)


    if len(matches) > MIN_MATCHES:
        print("Matches  found - %d/%d" % (len(matches), MIN_MATCHES))
        framePoints = np.float32([referenceImagePts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        destinationPoints = np.float32([framePts[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(framePoints, destinationPoints, cv2.RANSAC, 5.0)
        
        matchesMask = mask.ravel().tolist()

        height, width = referenceImage.shape
        corners = np.float32(
            [
                [0, 0],
                [0, height - 1],
                [width - 1, height - 1],
                [width - 1, 0]
            ]
        ).reshape(-1, 1, 2)
        transformedCorners = cv2.perspectiveTransform(corners, homography)
        frameMarker = cv2.polylines(frame, [np.int32(transformedCorners)], True, 255, 5, cv2.LINE_AA)

        drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        result = cv2.drawMatches(referenceImage, referenceImagePts, frameMarker, framePts, matches, None, **drawParameters)

        result = cv2.resize(result, (800, 400))
        cv2.imshow("result", result)
    else:
        print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCHES))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.concatenate((referenceImage, frame), axis=1)
        result = cv2.resize(result, (800, 400))
        cv2.imshow("result", result)

    key_input = cv2.waitKey(1) & 0xFF
    if key_input == "b":
        break

cv2.destroyAllWindows()