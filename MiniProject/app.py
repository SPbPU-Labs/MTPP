import cv2
import mediapipe as mp
import threading
import queue
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

frame_queue = queue.Queue(maxsize=1)
running = True


def calculate_distance(landmark1, landmark2):
    """Calculate distance between two points (landmarks)"""
    return math.sqrt(
        (landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2
    )


def producer():
    """Capture video frames and put them into the frame queue"""
    global running
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)
    cap.release()


def get_finger_state(landmarks):
    """Determine bend state for each finger (Thumb, Index, Middle, Ring, Pinky)"""
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    states = {}
    for name, tip, pip in zip(finger_names, finger_tips, finger_pips):
        if name == "Thumb":
            is_bent = landmarks.landmark[tip].x < landmarks.landmark[pip].x
        else:
            is_bent = landmarks.landmark[tip].y > landmarks.landmark[pip].y

        states[name] = "Bent" if is_bent else "Straight"

    return states


def detect_gesture(finger_states, landmarks):
    """Recognize gesture based on finger states and landmark positions"""
    thumb = finger_states["Thumb"]
    index = finger_states["Index"]
    middle = finger_states["Middle"]
    ring = finger_states["Ring"]
    pinky = finger_states["Pinky"]

    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Gesture "Victory" ✌️
    if (
        index == "Straight"
        and middle == "Straight"
        and ring == "Bent"
        and pinky == "Bent"
    ):
        return "Victory"

    # Gesture "Hello" 👋
    if (
        thumb == "Straight"
        and index == "Straight"
        and middle == "Straight"
        and ring == "Straight"
        and pinky == "Straight"
    ):
        return "Hello"

    # Gesture "OK" 👌
    if (
        thumb == "Straight"
        and index == "Bent"
        and calculate_distance(thumb_tip, index_tip) < 0.1
        and middle == "Straight"
        and ring == "Straight"
        and pinky == "Straight"
    ):
        return "OK"

    # Gesture "Thumbs Up" 👍
    if (
        thumb == "Straight"
        and index == "Bent"
        and middle == "Bent"
        and ring == "Bent"
        and pinky == "Bent"
    ):
        return "Thumbs Up"

    # Gesture "Rock" 🤘
    if (
        index == "Straight"
        and pinky == "Straight"
        and middle == "Bent"
        and ring == "Bent"
    ):
        return "Rock"

    # Gesture "Middle Finger" 🖕
    if middle == "Straight" and index == "Bent" and ring == "Bent" and pinky == "Bent":
        return "Middle Finger"

    return "Unknown"


def consumer():
    """Consume video frames from the frame queue and display them"""
    global running
    cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    finger_states = get_finger_state(hand_landmarks)

                    gesture = detect_gesture(finger_states, hand_landmarks)

                    y_offset = 50
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 40

                    for finger, state in finger_states.items():
                        cv2.putText(
                            frame,
                            f"{finger}: {state}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        y_offset += 30

            cv2.imshow("Hand Gesture Recognition", frame)

            if (
                cv2.waitKey(1) & 0xFF == ord("q")
                or cv2.getWindowProperty(
                    "Hand Gesture Recognition", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                running = False


if __name__ == "__main__":
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()
    cv2.destroyAllWindows()
