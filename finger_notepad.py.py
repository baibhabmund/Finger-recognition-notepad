import cv2
import numpy as np
import mediapipe as mp
import screeninfo

# Screen setup
screen = screeninfo.get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height
TOOLBAR_WIDTH = 180

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Color and tool settings
colors = [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0),
          (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0)]
color_index = 2
brush_sizes = [4, 10, 20]
eraser_sizes = [20, 40, 60]
brush_size = brush_sizes[1]
eraser_size = eraser_sizes[1]
selected_brush = 1
selected_eraser = 1
mode = "draw"
prev_x, prev_y = 0, 0

# UI layout
button_radius = 20
padding = 15
color_buttons = [(90, 40 + i * 60) for i in range(len(colors))]
brush_buttons = [(90, 500 + i * 60) for i in range(3)]
eraser_buttons = [(90, 700 + i * 60) for i in range(3)]
quit_button = (30, SCREEN_HEIGHT - 80, 150, SCREEN_HEIGHT - 20)

# Canvas and camera
canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)

cv2.namedWindow("Finger Notepad", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Finger Notepad", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def count_fingers(lmList):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = [int(lmList[tip_ids[0]][0] > lmList[tip_ids[0] - 1][0])]
    for i in range(1, 5):
        fingers.append(int(lmList[tip_ids[i]][1] < lmList[tip_ids[i] - 2][1]))
    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = [(int(lm.x * SCREEN_WIDTH), int(lm.y * SCREEN_HEIGHT)) for lm in handLms.landmark]
            fingers_up = count_fingers(lmList)
            x1, y1 = lmList[8]

            if x1 < TOOLBAR_WIDTH:
                for i, (cx, cy) in enumerate(color_buttons):
                    if (x1 - cx) ** 2 + (y1 - cy) ** 2 <= button_radius ** 2:
                        color_index = i
                        mode = 'draw'
                        prev_x, prev_y = 0, 0

                for i, (cx, cy) in enumerate(brush_buttons):
                    if abs(x1 - cx) < 30 and abs(y1 - cy) < 30:
                        brush_size = brush_sizes[i]
                        selected_brush = i
                        mode = 'draw'
                        prev_x, prev_y = 0, 0

                for i, (cx, cy) in enumerate(eraser_buttons):
                    if abs(x1 - cx) < 30 and abs(y1 - cy) < 30:
                        eraser_size = eraser_sizes[i]
                        selected_eraser = i
                        mode = 'erase'
                        prev_x, prev_y = 0, 0

                x1q, y1q, x2q, y2q = quit_button
                if x1q < x1 < x2q and y1q < y1 < y2q:
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            elif x1 > TOOLBAR_WIDTH:
                if fingers_up == 1:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1
                    thickness = eraser_size if mode == 'erase' else brush_size
                    color = (0, 0, 0) if mode == 'erase' else colors[color_index]
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), color, thickness)
                    prev_x, prev_y = x1, y1
                else:
                    prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.rectangle(output, (0, 0), (TOOLBAR_WIDTH, SCREEN_HEIGHT), (30, 30, 30), -1)

    for i, (cx, cy) in enumerate(color_buttons):
        cv2.circle(output, (cx, cy), button_radius, colors[i], -1)
        if i == color_index:
            cv2.circle(output, (cx, cy), button_radius + 4, (255, 255, 255), 2)

    for i, (cx, cy) in enumerate(brush_buttons):
        cv2.circle(output, (cx, cy), 10 + i * 4, (255, 255, 255), -1)
        if i == selected_brush and mode == 'draw':
            cv2.circle(output, (cx, cy), 16, (0, 255, 0), 2)

    for i, (cx, cy) in enumerate(eraser_buttons):
        size = 10 + i * 8
        cv2.rectangle(output, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (100, 100, 100), -1)
        if i == selected_eraser and mode == 'erase':
            cv2.rectangle(output, (cx - size // 2 - 4, cy - size // 2 - 4),
                          (cx + size // 2 + 4, cy + size // 2 + 4), (0, 0, 255), 2)

    x1q, y1q, x2q, y2q = quit_button
    cv2.rectangle(output, (x1q, y1q), (x2q, y2q), (0, 0, 255), -1)
    cv2.putText(output, "QUIT", (x1q + 30, y2q - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Finger Notepad", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
    elif key == ord('c'):
        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
