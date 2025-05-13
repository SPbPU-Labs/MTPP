import cv2
import mediapipe as mp
import threading
import queue
import math
import time
from argparse import ArgumentParser

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class HandGestureDetector:
    """Hand gesture detection class"""
    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def calculate_distance(self, landmark1, landmark2):
        """Calculate distance between two points (landmarks)"""
        return math.sqrt(
            (landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2
        )

    def get_finger_state(self, landmarks):
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

    def detect_gesture(self, finger_states, landmarks):
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
            and self.calculate_distance(thumb_tip, index_tip) < 0.1
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

    def process_frame(self, frame):
        """Process frame and return annotated frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
        ) as hands:
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    finger_states = self.get_finger_state(hand_landmarks)
                    gesture = self.detect_gesture(finger_states, hand_landmarks)

                    y_offset = 65
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

            # Calculate FPS
            self.frame_count += 1
            if time.time() - self.start_time >= 1.0:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.frame_count = 0
                self.start_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Mode: {'Multi-thread' if self.multi_thread else 'Single-thread'}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (222, 190, 73),
                2,
            )

            return frame

    def producer(self):
        """Capture video frames and put them into the frame queue"""
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put(frame)
        cap.release()

    def consumer(self):
        """Consume video frames from the frame queue and display them"""
        cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                frame = self.process_frame(frame)
                cv2.imshow("Hand Gesture Recognition", frame)

                if (
                    cv2.waitKey(1) & 0xFF == ord("q")
                    or cv2.getWindowProperty(
                        "Hand Gesture Recognition", cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    self.running = False
            else:
                time.sleep(0.01)

    def run_single_thread(self):
        """Run the application in a single thread"""
        cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Hand Gesture Recognition", frame)

            if (
                cv2.waitKey(1) & 0xFF == ord("q")
                or cv2.getWindowProperty(
                    "Hand Gesture Recognition", cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                self.running = False
        cap.release()
        cv2.destroyAllWindows()

    def run_multi_thread(self):
        """Run the application in multi-thread mode"""
        producer_thread = threading.Thread(target=self.producer)
        consumer_thread = threading.Thread(target=self.consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()
        cv2.destroyAllWindows()

    def run(self):
        """Run the application"""
        if self.multi_thread:
            self.run_multi_thread()
        else:
            self.run_single_thread()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--multi", action="store_true", help="Use multi-threading mode")
    args = parser.parse_args()

    detector = HandGestureDetector(multi_thread=args.multi)
    detector.run()
