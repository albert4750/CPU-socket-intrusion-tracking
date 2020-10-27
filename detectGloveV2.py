from imutils.video import FPS
import cv2


def main():
    cam = cv2.VideoCapture(0)  # video source
    tracker = cv2.TrackerCSRT_create()  # create tracker object
    initialBox = None  # box that we are tracking

    while True:
        # Capture frame
        ret, frame = cam.read()

        if initialBox is not None:  # if we are tracking something
            (success, box) = tracker.update(frame)  # find new coords of object
            if success:  # if successful
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fps.update()  # update fps
            fps.stop()
            info = [("Tracker", "CRST"), ("Success", "Yes" if success else "No"), ("FPS", "{:.2f}".format(fps.fps()))]
            # loop over the info and draw them on frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting
            # the ROI)
            initialBox = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput
            # estimator as well
            tracker.init(frame, initialBox)
            fps = FPS().start()  # initialize fps counter
        # print(str(ret))

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
