
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# -------------------- MEDIAPIPE SETUP --------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------- WEBCAM SETUP --------------------

def open_camera():
    for camera_index in [1, 0, 2]:
        camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

        if camera.isOpened():
            success, frame = camera.read()

            if success and frame is not None:
                print(f"Webcam opened successfully: camera {camera_index}")
                return camera

        camera.release()

    return None


cap = open_camera()

if cap is None:
    hands.close()
    raise SystemExit(
        "Could not open webcam. Check camera permissions and close other apps using it."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------- GAME VARIABLES --------------------

choices = ["Rock", "Paper", "Scissors"]

player_score = 0
computer_score = 0

player_move = "Waiting..."
computer_move = "Waiting..."
message = "Show your hand to play!"

last_gesture = None
gesture_start = time.time()

round_active = False
round_time = 0

# -------------------- HAND GESTURE DETECTION --------------------

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark

    # Index, middle, ring and pinky fingers
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    extended = []

    for tip, joint in zip(tips, joints):
        extended.append(
            landmarks[tip].y < landmarks[joint].y
        )

    # All four fingers extended = Paper
    if all(extended):
        return "Paper"

    # Index and middle extended = Scissors
    if extended == [True, True, False, False]:
        return "Scissors"

    # All four fingers folded = Rock
    if not any(extended):
        return "Rock"

    return None


# -------------------- COMPUTER MOVE DRAWING --------------------

def draw_computer_move(panel, move):
    center_x = panel.shape[1] // 2
    center_y = 230

    if move == "Rock":
        # Draw a rock
        points = np.array([
            [center_x - 65, center_y - 10],
            [center_x - 45, center_y - 55],
            [center_x + 10, center_y - 70],
            [center_x + 55, center_y - 35],
            [center_x + 65, center_y + 25],
            [center_x + 25, center_y + 60],
            [center_x - 40, center_y + 50],
            [center_x - 65, center_y + 15]
        ], dtype=np.int32)

        cv2.fillPoly(panel, [points], (110, 110, 125))
        cv2.polylines(panel, [points], True, (65, 65, 80), 4)

        cv2.putText(
            panel, "ROCK",
            (center_x - 65, center_y + 105),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (70, 70, 80), 3
        )

    elif move == "Paper":
        # Draw an open sheet of paper
        x1, y1 = center_x - 65, center_y - 90
        x2, y2 = center_x + 65, center_y + 85

        cv2.rectangle(panel, (x1, y1), (x2, y2), (235, 235, 245), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (80, 100, 200), 3)

        for y in range(y1 + 35, y2 - 10, 30):
            cv2.line(
                panel, (x1 + 18, y), (x2 - 18, y),
                (170, 175, 215), 2
            )

        cv2.putText(
            panel, "PAPER",
            (center_x - 65, center_y + 125),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (80, 100, 200), 3
        )

    elif move == "Scissors":
        # Draw a pair of scissors
        cv2.line(
            panel, (center_x - 10, center_y + 10),
            (center_x - 60, center_y - 65), (40, 70, 220), 10
        )
        cv2.line(
            panel, (center_x + 10, center_y + 10),
            (center_x + 60, center_y - 65), (40, 70, 220), 10
        )
        cv2.line(
            panel, (center_x - 10, center_y + 10),
            (center_x + 50, center_y + 70), (40, 70, 220), 10
        )
        cv2.line(
            panel, (center_x + 10, center_y + 10),
            (center_x - 50, center_y + 70), (40, 70, 220), 10
        )

        cv2.circle(panel, (center_x - 60, center_y - 65), 23, (40, 70, 220), 6)
        cv2.circle(panel, (center_x + 60, center_y - 65), 23, (40, 70, 220), 6)
        cv2.circle(panel, (center_x, center_y + 10), 10, (30, 30, 30), -1)

        cv2.putText(
            panel, "SCISSORS",
            (center_x - 90, center_y + 125),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (40, 70, 220), 3
        )


# -------------------- MAIN GAME LOOP --------------------

try:
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        # Process hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks)

        now = time.time()

        # Track a stable gesture before playing
        if gesture != last_gesture:
            last_gesture = gesture
            gesture_start = now

        # Play when the gesture remains stable for 0.7 seconds
        if (
            gesture is not None
            and not round_active
            and now - gesture_start >= 0.7
        ):
            player_move = gesture
            computer_move = random.choice(choices)

            if player_move == computer_move:
                message = "DRAW!"

            elif (
                (player_move == "Rock" and computer_move == "Scissors")
                or (player_move == "Paper" and computer_move == "Rock")
                or (player_move == "Scissors" and computer_move == "Paper")
            ):
                message = "YOU WIN!"
                player_score += 1

            else:
                message = "COMPUTER WINS!"
                computer_score += 1

            round_active = True
            round_time = now

        # Show the result for two seconds
        if round_active and now - round_time >= 2:
            round_active = False
            message = "Show your next move!"

        # -------------------- LEFT PANEL: PLAYER --------------------

        camera_panel = cv2.resize(frame, (480, 500))

        cv2.rectangle(camera_panel, (0, 0), (480, 55), (35, 35, 35), -1)

        cv2.putText(
            camera_panel, "YOU",
            (190, 38), cv2.FONT_HERSHEY_SIMPLEX,
            1, (80, 230, 120), 2
        )

        cv2.putText(
            camera_panel, f"Move: {gesture or 'Detecting...'}",
            (15, 410), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.putText(
            camera_panel, f"Score: {player_score}",
            (15, 455), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (80, 230, 120), 2
        )

        # -------------------- RIGHT PANEL: COMPUTER --------------------

        computer_panel = np.full(
            (500, 480, 3), (242, 242, 250), dtype=np.uint8
        )

        cv2.rectangle(computer_panel, (0, 0), (480, 55), (35, 35, 35), -1)

        cv2.putText(
            computer_panel, "COMPUTER",
            (125, 38), cv2.FONT_HERSHEY_SIMPLEX,
            1, (100, 180, 255), 2
        )

        cv2.putText(
            computer_panel, f"Score: {computer_score}",
            (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (30, 80, 180), 2
        )

        draw_computer_move(computer_panel, computer_move)

        # Display the round result
        cv2.putText(
            computer_panel, message,
            (35, 430), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (40, 130, 40), 2
        )

        # -------------------- COMBINE BOTH PANELS --------------------

        screen = np.hstack((camera_panel, computer_panel))

        cv2.putText(
            screen, "Q: Quit | R: Reset Score",
            (15, 490), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 255), 1
        )

        cv2.imshow("Rock Paper Scissors Game", screen)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            player_score = 0
            computer_score = 0
            player_move = "Waiting..."
            computer_move = "Waiting..."
            message = "Score reset!"
            round_active = False
            last_gesture = None
            gesture_start = time.time()

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Game closed.")
