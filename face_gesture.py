
import cv2
import numpy as np

# Face Detection Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

GESTURES = {
    0: "Fist",
    1: "Peace (2 fingers)",
    2: "Three fingers",
    3: "Four fingers",
    4: "Open Palm (5)",
}


def detect_gesture(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([0, 30, 60]),
        np.array([20, 150, 255])
    )

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "No hand", mask

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 3000:
        return "No hand", mask

    hull = cv2.convexHull(cnt, returnPoints=False)

    if hull is None or len(hull) < 3:
        return "Unknown", mask

    defects = cv2.convexityDefects(cnt, hull)

    count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = cnt[s][0]
            end = cnt[e][0]
            far = cnt[f][0]

            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)

            if b * c == 0:
                continue

            angle = np.arccos(
                np.clip(
                    (b*b + c*c - a*a) / (2*b*c),
                    -1,
                    1
                )
            )

            if angle <= np.pi / 2 and d > 8000:
                count += 1

    return GESTURES.get(min(count, 4), "Unknown"), mask


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("Face and Gesture Detection Started!")
    print("Press Q to quit.")
    print("Press S to save screenshot.")

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Error: Cannot read webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            1.2,
            5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Hand Gesture Detection
        height, width = frame.shape[:2]

        x1 = min(400, width - 1)
        y1 = min(80, height - 1)
        x2 = min(620, width)
        y2 = min(300, height)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            gesture, mask = detect_gesture(roi)
        else:
            gesture = "No hand"
            mask = np.zeros((100, 100), dtype=np.uint8)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Show hand here",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2
        )

        # Display Windows
        cv2.imshow("Face & Gesture Detection", frame)
        cv2.imshow("Hand Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            cv2.imwrite("screenshot.png", frame)
            print("Saved screenshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
