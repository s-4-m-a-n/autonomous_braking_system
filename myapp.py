# from json import load
# from math import factorial
import torch
import numpy as np
from time import time
import cv2
import json_loader
from datetime import datetime
import imutils

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
        AVG_CAR_WIDTH,\
        CONFIDENCE, BOTTLE_HEIGHT, OBJECT_MEASURED_DISTANCE,\
            BRAKING_DISTANCE, ALERT_DISTANCE,TEST_IMAGE_WIDTH,\
                TEST_IMAGE_HEIGHT, ROI_OFFSET_DISTANCE

    FOCAL_LENGTH = config["focal_length"]
    AVG_CAR_HEIGHT = config["avg_heights"]["car"]
    AVG_BIKE_HEIGHT = config["avg_heights"]["bike"]
    CONFIDENCE = config["confidence"]
    BOTTLE_HEIGHT = config["avg_heights"]["bottle"] #INCH
    OBJECT_MEASURED_DISTANCE = config["object_measured_distance"] #INCH
    ALERT_DISTANCE = config['thresholds']['alert']['distance']
    BRAKING_DISTANCE = config['thresholds']['emergency_braking']['distance']
    TEST_IMAGE_WIDTH = config['img_width']
    TEST_IMAGE_HEIGHT = config['img_height']
    AVG_CAR_WIDTH = config['avg_width']['car']
    ROI_OFFSET_DISTANCE = -12 # inch


def find_ROI(f_width, f_height, traffic_objs):
    bboxes = {}
    for obj in traffic_objs:
        for _ , param in obj.items():
            
            distance = param["distance"] + ROI_OFFSET_DISTANCE
            roi_pix_width = FOCAL_LENGTH * AVG_CAR_WIDTH / distance 
        
            center_x = f_width//2
            roi_x1 = center_x - roi_pix_width//2

            bbox = int(roi_x1), 0 , int(roi_pix_width), f_height
            bboxes[param["distance"]] = bbox

    return bboxes



def plot_ROI(bbox,frame):
    # plot only the nearest bbox
    distances = [key for key in bbox.keys()]
    min_dist = sorted(distances)[0]

    x, y, w, h = bbox[min_dist][0], bbox[min_dist][1], bbox[min_dist][2], bbox[min_dist][3]
  
    shapes = np.zeros_like(frame, np.uint8)

    cv2.rectangle(shapes, (x,y), (x+w, y+h), (0,0,255), cv2.FILLED)
    #mask 
    out = frame.copy()
    alpha = 0.3
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    return out



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
            cv2.putText(frame,label[obj_index],(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255),1)
    return frame




def get_traffic_object_info(frame_shape, results, class_names):
    label_index, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    label = [class_names[int(index)] for index in label_index]
    f_width, f_height = frame_shape
    traffic_objs = []
  
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
            distance = (AVG_BIKE_HEIGHT * FOCAL_LENGTH )/pix_height #  make it dynamic so that it can work for other object also

            print(f"{label[obj_index]} distance : {distance}")

            traffic_objs.append({label[obj_index] : {"distance":distance, "x_left_pos": x1, "x_right_pos": x2 }})
    return traffic_objs


def trigger_braking_signal(traffic_objs, roi_bbox):
    for obj in traffic_objs:
        for name, param in obj.items():
            if param['distance'] <= BRAKING_DISTANCE:
                print("distance",param['distance'])
                condition_first = param['x_left_pos'] >= roi_bbox[param['distance']][0] and param['x_left_pos'] <= roi_bbox[param['distance']][2]
                condition_second = param['x_right_pos'] >= roi_bbox[param['distance']][0] and param['x_right_pos'] <= roi_bbox[param['distance']][2]
                condition_third =param['x_left_pos'] <= roi_bbox[param['distance']][0] and param['x_right_pos'] >= roi_bbox[param['distance']][2]

                # print(param['x_left_pos'],roi_bbox[2])
                if condition_first or condition_second or condition_third:
                    print(f"!! send braking signal !! : {name} @ distance: {param['distance']}")   


def run():
    
    #loading yolo v5 model from pytorch hub
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    #loading model locally
    model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5s.pt', source='local')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)


    # loading source
    source = cv2.VideoCapture(TEST_VIDEO_PATH)
    win_name = "yolov5 autonomous braking system"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


    # writing file
    # f_width , f_height = int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),\
                                #  int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print("test image shape ",f_width, f_height)
    f_width, f_height = TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT

    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height)) 
    
    

    while cv2.waitKey(1) != 27:
        start_time = time()
        curr_time = datetime.now().strftime("%H:%M:%S")
        print(f"================{curr_time}========================")
        ok, frame = source.read()

        frame = imutils.resize(frame, width=f_width)

        if not ok:
            print("error")
            break

        # object detection using pretrained model
        # coord is in the form of xmin, ymin, xmax. ymax, confidence
        results = model([frame])
        
        
        # draw rectangle bounding box
        b_frame = plot_bounding_box(frame, results, model.names)

        traffic_objs = get_traffic_object_info((frame.shape[1], frame.shape[0]), results,model.names,)

        # estimating 
        roi_bboxes = find_ROI(f_width, f_height, traffic_objs)

        print(f"\nroi_bboxes {roi_bboxes}\n")

        # focal_length = focal_length_estimator(frame)
        trigger_braking_signal(traffic_objs, roi_bboxes)

        
        #draw roi in the frame
        b_frame = plot_ROI(roi_bboxes, b_frame)

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
