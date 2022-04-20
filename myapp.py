
import torch
from time import time
import cv2
import json_loader
from datetime import datetime
import imutils
from helper import find_ROI, plot_ROI, plot_bounding_box,\
                    get_traffic_object_info
from colored import fg, bg, attr
# config
# FOCAL_LENGTH = 240
# AVG_CAR_HEIGHT = 150
# AVG_BIKE_HEIGHT = 55 # INCH 120 distance 
TEST_VIDEO_PATH = "../test videos/101.mp4"
CONFIG_FILE_PATH = "config.json"
# CONFIDENCE = 0.4
# BOTTLE_HEIGHT = 15 #INCH
# OBJECT_MEASURED_DISTANCE = 24 #INCH


def get_config(file_name):
    config = json_loader.load_config(file_name)
    global FOCAL_LENGTH, CONFIDENCE, OBJECT_MEASURED_DISTANCE,\
        BRAKING_DISTANCE, ALERT_DISTANCE, TEST_IMAGE_WIDTH,\
        TEST_IMAGE_HEIGHT, ROI_OFFSET_DISTANCE, AVG_OBJS_HEIGHT, MY_WIDTH

    FOCAL_LENGTH = config["focal_length"]
    CONFIDENCE = config["confidence"]

    OBJECT_MEASURED_DISTANCE = config["object_measured_distance"]  # INCH

    ALERT_DISTANCE = config['thresholds']['alert']['distance']
    BRAKING_DISTANCE = config['thresholds']['emergency_braking']['distance']

    TEST_IMAGE_WIDTH = config['img_width']
    TEST_IMAGE_HEIGHT = config['img_height']

    ROI_OFFSET_DISTANCE = -12  # inch

    AVG_OBJS_HEIGHT = config['avg_heights']
    MY_WIDTH = config['avg_width']['car']  # change the vehicle type




def trigger_braking_signal(traffic_objs, roi_bbox):
    for obj in traffic_objs:
        for name, param in obj.items():
            if param['distance'] <= BRAKING_DISTANCE:
                print("distance", param['distance'])
                condition_first = param['x_left_pos'] >= roi_bbox[param['distance']][0] and param['x_left_pos'] <= roi_bbox[param['distance']][2]
                condition_second = param['x_right_pos'] >= roi_bbox[param['distance']][0] and param['x_right_pos'] <= roi_bbox[param['distance']][2]
                condition_third = param['x_left_pos'] <= roi_bbox[param['distance']][0] and param['x_right_pos'] >= roi_bbox[param['distance']][2]

                # print(param['x_left_pos'],roi_bbox[2])
                if condition_first or condition_second or condition_third:
                    color = fg("white") + bg("red")
                    print(color + f"!! send braking signal !! : {name} @ distance: {param['distance']}"+ attr("reset"))   


def run():
    
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
    frame_counter = 0
    while cv2.waitKey(1) != 27:
        start_time = time()
        curr_time = datetime.now().strftime("%H:%M:%S")
        frame_counter += 1
        
        color = fg("yellow")  
        print(color + f"\n\n================{curr_time}  frame: {frame_counter}\
            =========================================")
        print("===============================================================\
                ======================"+attr("reset"))

        ok, frame = source.read()
        frame = imutils.resize(frame, width=f_width)

        if not ok:
            print("error")
            break

        # object detection using pretrained model
        # coord is in the form of xmin, ymin, xmax. ymax, confidence
        results = model([frame])

        # draw rectangle bounding box
        b_frame = plot_bounding_box(frame, results, model.names, CONFIDENCE)

        traffic_objs = get_traffic_object_info(
                            (frame.shape[1], frame.shape[0]),
                            results, model.names, CONFIDENCE,
                            AVG_OBJS_HEIGHT, FOCAL_LENGTH)

        # estimating
        roi_bboxes = find_ROI(f_width, f_height, traffic_objs,  FOCAL_LENGTH,
                              ROI_OFFSET_DISTANCE, MY_WIDTH)

        print(fg(5)+f"\n .......roi_bboxes......\n {roi_bboxes}\n ............\
                ..................." + attr("reset"))

        trigger_braking_signal(traffic_objs, roi_bboxes)

        # draw roi in the frame
        b_frame = plot_ROI(roi_bboxes, b_frame)

        end_time = time()
        fps = 1 / round(end_time - start_time, 3)
        # out.write(b_frame)
        print(fg("white")+bg(30) + f"fps : " + attr("reset") + f"{fps}")
        cv2.imshow(win_name, b_frame)

    source.release()
    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    get_config(CONFIG_FILE_PATH)
    run()
