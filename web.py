import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# HSV range for blue marker
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

draw_color = (255, 0, 0)  # default blue
thickness = 8
prev_x, prev_y = None, None

def draw_ui(img):
    cv2.rectangle(img, (0, 0), (1280, 80), (50, 50, 50), -1)

    cv2.putText(img, "AIR CANVAS", (520, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Buttons
    cv2.rectangle(img, (50, 20), (130, 60), (255, 0, 0), -1)    # Blue
    cv2.rectangle(img, (150, 20), (230, 60), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (250, 20), (330, 60), (0, 0, 255), -1)  # Red
    cv2.rectangle(img, (350, 20), (450, 60), (0, 0, 0), -1)    # Eraser
    cv2.rectangle(img, (480, 20), (600, 60), (255, 255, 255), -1)

    cv2.putText(img, "CLR", (510, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    draw_ui(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            (x, y), r = cv2.minEnclosingCircle(c)
            x, y = int(x), int(y)

            if y < 80:
                if 50 < x < 130:
                    draw_color = (255, 0, 0)
                elif 150 < x < 230:
                    draw_color = (0, 255, 0)
                elif 250 < x < 330:
                    draw_color = (0, 0, 255)
                elif 350 < x < 450:
                    draw_color = (0, 0, 0)
                elif 480 < x < 600:
                    canvas = np.zeros_like(canvas)
            else:
                if prev_x is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Advanced Air Canvas", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
