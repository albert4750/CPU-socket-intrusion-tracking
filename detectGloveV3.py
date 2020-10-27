import cv2
import imutils
import numpy as np
from imutils.video import FPS
import time


def detectAndDescribe(image, descriptor):  # for finding key points and features of the key points
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (keypoints, features) = descriptor.detectAndCompute(gray, None)
    keypoints = np.float32([kp.pt for kp in keypoints])

    return keypoints, features


def matchKeypoints(kpts, tKpts, f, tF, ratio=0.75):  # find keypoints that matches
    if f is None or tF is None:
        return None, None
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(f, tF, 2)
    matches = []

    for match in rawMatches:
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            matches.append((match[0].trainIdx, match[0].queryIdx))

    pts = np.float32([kpts[i] for (_, i) in matches])
    tPts = np.float32([tKpts[i] for (i, _) in matches])
    return pts, tPts


def main():
    cam = cv2.VideoCapture(0)  # video source
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set to full HD as it increases accuracy
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    sift = cv2.SIFT_create()  # create our feature finding object

    fps = FPS().start()  # start FPS tracker

    template = cv2.imread("cpu1080new.png")  # load in template
    kps, feature = detectAndDescribe(template, sift)  # find key points and features of template

    delay = 0  # delay is used for knowing how many frames have we passed since motherboard was placed
    pointsX = []  # list that we use to store coordinates of key points
    pointsY = []
    corners = [(0, 0), (0, 0)]  # corners of the box we found stored in this list
    f2Use = 10  # frames to use for finding key points
    curr = 0
    timetook = []

    found = False  # boolean of whether we have a found the cpu socket

    warning = np.zeros((400, 400, 3), np.uint8)  # a warning window that would change color depending on condition

    while True:
        ret, frame = cam.read()  # get frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret is used to determined whether a motherboard have been placed

        if ret > 135:  # increment to know how much frame passed since mb placed
            delay += 1
        elif ret < 135 and delay != 0:  # reset everything once mb was removed
            delay = 0
            pointsX = []
            pointsY = []
            corners = [(0, 0), (0, 0)]
            curr = 0
            found = False

        if 5 <= delay <= f2Use + 5:  # we collect key points after mb have been place for several frames
            if delay == 5:
                curr = time.time_ns()
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 5)
            sframe = frame[y:y+(h//2), int(x * 1.3):x+int(w/1.5)]  # sframe stand for smaller frame, we are assuming
            # that cpu socket would be within a general area on mb

            tKps, tFeature = detectAndDescribe(sframe, sift)  # find keypoints and features on frame

            pts, tPts = matchKeypoints(kps, tKps, feature, tFeature)  # match keypoints

            if tPts is not None:
                for p in tPts:
                    cv2.circle(sframe, (p[0], p[1]), 4, (0, 0, 255), -1)  # red circles are the key points that matched
                    pointsX.append(int(p[0]))  # store xy coordinates
                    pointsY.append(int(p[1]))

        elif delay == f2Use + 6 and len(pointsX) != 0 and len(pointsY) != 0:  # find box at this specific frame
            # we are assuming that the board wouldn't move once placed
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key=cv2.contourArea)
            (X, Y, w, h) = cv2.boundingRect(c)

            x = np.array(pointsX)  # convert points recorded to np array
            y = np.array(pointsY)
            x = x[abs(x - np.mean(x)) < (1.5 * np.std(x))]  # only keep data within certain standard deviation
            y = y[abs(y - np.mean(y)) < (1.5 * np.std(y))]

            corners = [(x.min() + int(X * 1.3), y.min() + Y), (x.max() + int(X * 1.3), y.max() + Y)]  # find corners pos
            # print(corners)

            found = True  # set that we found the box

            timetook.append((time.time_ns() - curr)/(10**9))

        if found:
            cpuF = frame[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]  # crop a frame for the cpu box found

            area = cpuF.shape[0] * cpuF.shape[1]  # amount of pixels in box as "area"

            cv2.rectangle(frame, corners[0], corners[1], 255, 2)  # draw on main frame

            if cpuF is not None:
                cv2.imshow("cpuF", cpuF)

            # now we draw two more frames, one showing the orange in the frame and black in frame
            # orange is used to detect gloves getting into the frame, and black for whether cover closed
            orange = cv2.inRange(cpuF, np.array([10, 20, 110], dtype="uint8"), np.array([80, 120, 250], dtype="uint8"))
            black = cv2.inRange(cpuF, np.array([0, 0, 0], dtype="uint8"), np.array([32, 32, 32], dtype="uint8"))
            # white = cv2.inRange(cpuF, np.array([90, 90, 90], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))

            cv2.imshow("orange", orange)
            cv2.imshow("black", black)

            pBlack = cv2.countNonZero(black)/area
            pOrange = cv2.countNonZero(orange)/area
            # pWhite = cv2.countNonZero(white)/area

            # show the percentage each color takes up
            # print("Percent black: %.3f; Percent orange: %.3f" % (pBlack, pOrange))
            # print("Percent black: %.3f; Percent orange: %.3f; Percent white: %.3f" % (pBlack, pOrange, pWhite))

            # if the color covers certain percentage then perform certain actions
            if pBlack > .3:  # if more than 30% black then assume lid closed, so we stop checking as if nothing found
                found = False
            if pOrange > .02:  # if more than 2% orange then show warning, in this case the warning frame is red
                warning[:, :] = (0, 0, 255)
                print("warning")
            else:  # else warning frame is green
                warning[:, :] = (0, 255, 0)

        fps.update()
        fps.stop()

        show = cv2.resize(frame, (1280, 720))
        cv2.putText(show, "FPS: " + "{:.2f}".format(fps.fps()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.imshow("warning", warning)
        cv2.imshow("frame", show)
        # cv2.imshow("gray", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # elif cv2.waitKey(1) & 0xFF == ord('s'):
        # print(str(ret))

    cam.release()
    cv2.destroyAllWindows()
    print("Time took for finding socket averages to %.3fs per try with %d tries" % (sum(timetook)/len(timetook),
                                                                                    len(timetook)))
    print(timetook)


if __name__ == "__main__":
    main()
