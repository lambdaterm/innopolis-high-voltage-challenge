import cv2

# reading the input
cap = cv2.VideoCapture(r'D:\PyProjects\innopolis-high-voltage-challenge\data\video\500_vertical_1[1920x1090_60fps].MP4')

out = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))

while (True):
    ret, frame = cap.read()
    if (ret):
        # writing the new frame in output
        out.write(frame)
        # cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 3)
        cv2.imshow("output", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            out.release()
            break

cv2.destroyAllWindows()
out.release()
cap.release()