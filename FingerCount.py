import HandTrackingModule as htm
import cv2
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class FingerCounter:
    def __init__(self):
        self.finger_count = 0

    def count_fingers(self, lm_list):
        self.finger_count = 0
        if lm_list is not None and len(lm_list) != 0:
            if lm_list[8][2] < lm_list[7][2]:
                self.finger_count += 1
            if lm_list[12][2] < lm_list[11][2]:
                self.finger_count += 1
            if lm_list[16][2] < lm_list[15][2]:
                self.finger_count += 1
            if lm_list[20][2] < lm_list[19][2]:
                self.finger_count += 1
            if lm_list[4][1] > lm_list[20][1]:
                if lm_list[4][1] > lm_list[3][1]:
                    self.finger_count += 1
            else:
                if lm_list[4][1] < lm_list[3][1]:
                    self.finger_count += 1

        return self.finger_count

    def get_raised_fingers(self, lm_list):
        raised_fingers = []
        if lm_list is not None and len(lm_list) != 0:
            if lm_list[4][1] > lm_list[20][1]:
                if lm_list[4][1] > lm_list[3][1]:
                    raised_fingers.append(0)
            else:
                if lm_list[4][1] < lm_list[3][1]:
                    raised_fingers.append(0)
            if lm_list[8][2] < lm_list[7][2]:
                raised_fingers.append(1)
            if lm_list[12][2] < lm_list[11][2]:
                raised_fingers.append(2)
            if lm_list[16][2] < lm_list[15][2]:
                raised_fingers.append(3)
            if lm_list[20][2] < lm_list[19][2]:
                raised_fingers.append(4)

            return raised_fingers


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()
    finger_counter = FingerCounter()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, [4, 8, 12, 16, 20])

        finger = finger_counter.count_fingers(lm_list)
        raised_fingers = finger_counter.get_raised_fingers(lm_list)

        cv2.putText(img, f'{str(finger)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, f'{str(raised_fingers)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
