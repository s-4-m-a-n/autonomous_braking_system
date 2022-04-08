from json import load
from math import factorial
import torch
import numpy as np
from time import time
import cv2
import json_loader

# config
# FOCAL_LENGTH = 240
# AVG_CAR_HEIGHT = 150
# AVG_BIKE_HEIGHT = 55 # INCH 120 distance 
TEST_VIDEO_PATH = "../test videos/103.mp4"
CONFIG_FILE_PATH = "config.json"
# CONFIDENCE = 0.4
# BOTTLE_HEIGHT = 15 #INCH
# OBJECT_MEASURED_DISTANCE = 24 #INCH


def get_config(file_name):
    config = json_loader.load_config(file_name)
    global FOCAL_LENGTH, AVG_CAR_HEIGHT, AVG_BIKE_HEIGHT,\
        CONFIDENCE, BOTTLE_HEIGHT, OBJECT_MEASURED_DISTANCE

    FOCAL_LENGTH = config["focal_length"]
    AVG_CAR_HEIGHT = config["avg_heights"]["car"]
    AVG_BIKE_HEIGHT = config["avg_heights"]["bike"]
    CONFIDENCE = config["confidence"]
    BOTTLE_HEIGHT = config["avg_heights"]["bottle"] #INCH
    OBJECT_MEASURED_DISTANCE = config["object_measured_distance"] #INCH


def plot_bounding_box(frame, results, class_names):
    label_index, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    label = [class_names[int(index)] for index in label_index]

    n = len(label)

    f_width, f_height = frame.shape[1], frame.shape[0]
    
    for obj_index in range(n):
        row = coord[obj_index]
        if row[4] >= CONFIDENCE:
            x1, y1, x2, y2 = int(row[0]*f_width), int(row[1]*f_height),\
                             int(row[2]*f_width), int(row[3]*f_height), 

            color = (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label[obj_index],(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),1)
    return frame




def distance_measure(frame_shape, results, class_names):
    label_index, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    label = [class_names[int(index)] for index in label_index]
    f_width, f_height = frame_shape
    distances = []
    for obj_index in range(len(label)):
        row  = coord[obj_index]
        if row[4] >= CONFIDENCE:
            x1, y1, x2, y2 = int(row[0]*f_width), int(row[1]*f_height),\
                             int(row[2]*f_width), int(row[3]*f_height), 
            pix_height = float(y2-y1)
            # FOCAL_LENGTH = (pix_height * OBJECT_MEASURED_DISTANCE) / BOTTLE_HEIGHT
            # print("----------------FOCAL LENGTH:-------------",FOCAL_LENGTH)
            # coeff = 110 / pix_height
            # print("coeff",coeff)
            coeff = 1.01
            distance = (AVG_BIKE_HEIGHT * FOCAL_LENGTH )/(coeff*pix_height)
            distances.append({label[obj_index] : distance})
    return distances



def run():
    
    #loading yolo v5 model from pytorch hub
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    #loading model locally
    model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5s.pt', source='local')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)


    # loading source
    source = cv2.VideoCapture(TEST_VIDEO_PATH)
    win_name = "yolo v5 obj detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # writing file
    # frame_width , frame_height = int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),\
    #                              int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height)) 


    while cv2.waitKey(1) != 27:
        start_time = time()
        ok, frame = source.read()

        if not ok:
            print("error")
            break

        # object detection using pretrained model
        # coord is in the form of xmin, ymin, xmax. ymax, confidence
        results = model([frame])
        
        # draw rectangle bounding box
        b_frame = plot_bounding_box(frame, results, model.names)
        # focal_length = focal_length_estimator(frame)
        distances = distance_measure((frame.shape[0], frame.shape[1]), results,model.names)
        print(distances)

        end_time = time()
        fps = 1 / round(end_time - start_time, 3) 
        # out.write(b_frame)
        print(f"fps : {fps}")
        cv2.imshow(win_name, b_frame)

    source.release()
    cv2.destroyWindow(win_name)



if __name__ == "__main__":
    get_config(CONFIG_FILE_PATH)
    # print(CONFIDENCE)
    run()