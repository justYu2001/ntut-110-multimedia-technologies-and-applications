import cv2
import os
import glob
import random

data_path = 'frames'
input_video_filename = 'helmet1.mp4'
output_video_filename = 'output.mp4'

output_video_fps = 10
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

cap = cv2.VideoCapture(input_video_filename)
video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
cap.release()

out_vid = cv2.VideoWriter(output_video_filename, fourcc, output_video_fps, (1280,  720))

classes = open(f'{data_path}/classes.txt.').read().strip().split()
random.seed(0)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

label_files = list(glob.glob(os.path.join(data_path, "[!classes]*.txt")))

for label_filename in label_files:
    labels  = open(label_filename).read().splitlines()
    image_name, _ = os.path.splitext(label_filename)
    image_name += '.png'
    image = cv2.imread(image_name)
    height, width, channels = image.shape
    for label in labels:
        element = label.split()
        x_center, y_center, w, h = float(element[1])*width, float(element[2])*height, float(element[3])*width, float(element[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2) 

        cv2.rectangle(image, (x1, y1), (x2, y2), colors[int(element[0])], thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, classes[int(element[0])], (x1, y1), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        out_vid.write(image)

cv2.imshow(image_name,image)
out_vid.release()
cv2.waitKey(0)
cv2.destroyAllWindows()