import cv2
import imutils


def main():
    cam = cv2.VideoCapture(0)  # video source
    cpuTemp = cv2.imread("withbracket.png")  # give template
    W = cpuTemp.shape[1]  # size of template which we will use to draw box later
    H = cpuTemp.shape[0]

    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Our operations on the frame come here

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret is the value that THRESH_OTSU uses for threshold, it changes as motherboard was placed

        if ret > 130:  # we only do calculations when motherboard was placed to save computing resources
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)  # find contour of motherboard and draw it out

            res = cv2.matchTemplate(frame, cpuTemp, cv2.TM_CCOEFF)  # match template
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            x = top_left[0]
            y = top_left[1]

            cv2.rectangle(frame, (x, y), (x + W, y + H), 255, 2)  # draw template box

        # Display the resulting frame
        cv2.imshow("frame: ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
