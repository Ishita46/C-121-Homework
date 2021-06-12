import cv2
import time
import numpy as np
from numpy.core.fromnumeric import resize

frame = cv2.resize(frame, (640, 480))
image = cv2.resize(image, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    u_black = np.array([0, 120, 50])
    l_black = np.array([10, 255, 255])

    mask = cv2.inRange(frame, u_black, l_black)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    f = frame - res
    f = np.where(f == 0, image, f)

    final_output = cv2.addWeighted(res, 1, 0)
    output_file.write(final_output)

    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows