import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

# get windows
# cap.set(3,640)
# cap.set(4,480)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # get Hand1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # hand type Left or Right

        # print(handType1)
        fingers1 = detector.fingersUp(hand1)
        cnt = fingers1[0]+fingers1[1]+fingers1[2]+fingers1[3]+fingers1[4]
        # print("Num_Fingers = ", cnt)
        # orders
        if cnt==1:
            if handType1=='Right':
                print("Turn Right")
            elif handType1=='Left':
                print("Turn Turn Left")
        elif cnt==0:
            print("Go!")
        elif cnt==5:
            print("Stop!")


        # # if we want to add orders(Use Two Hands)
        # if len(hands==2):
        #     hand2 = hands[1]
        #     lmlist2 = hand2["lmlist"]
        #     bbox2 = hand2["bbox"]
        #     centerPoint2 = hand2["center"]
        #     handType2 = hand2["type"]
        #
        #     fingers2 = detector.fingersUp(hand2)
        #     print("----------Two Hands--------")


    cv2.imshow("Hands", img)
    cv2.waitKey(1)