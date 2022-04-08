import cv2
import torch
import json
# config
CONFIG_IMAGE_PATH = "config image/test.jpg"
OBJECT_REAL_HEIGHT = 15 #inch
OBJECT_MEASURED_DISTANCE = 24 #inch
CONFIDENCE = 0.4
CONFIG_FILE_PATH = "config.json" 

def get_focal_length(frame_shape, results):
    global FOCAL_LENGTH 

    coord = results.xyxyn[0][:, :-1]
    f_width, f_height = frame_shape

    row  = coord[0]
    
    if row[4] >= CONFIDENCE:
        x1, y1, x2, y2 = int(row[0]*f_width), int(row[1]*f_height),\
                            int(row[2]*f_width), int(row[3]*f_height), 
        pix_height = float(y2-y1)
        FOCAL_LENGTH = (pix_height * OBJECT_MEASURED_DISTANCE) / OBJECT_REAL_HEIGHT        
    else:
        print("is object is detected")
        FOCAL_LENGTH = -1

    return FOCAL_LENGTH




def run():
    #loading model locally
    model = torch.hub.load(r'yolov5', 'custom', path=r'yolov5s.pt', source='local')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # load image
    image  = cv2.imread(CONFIG_IMAGE_PATH)

    image_shape = image.shape[0], image.shape[1]

    results  = model([image])

    focal_length = get_focal_length(image_shape, results)

    
    if focal_length != -1:
        # load config json
        with open(CONFIG_FILE_PATH,"r") as f:
            config = json.load(f)

        # updating config params
        config["focal_length"] = focal_length
        config['confidence'] = CONFIDENCE
        config["object_measured_distance"] = OBJECT_MEASURED_DISTANCE
        config["avg_heights"]["bottle"] = OBJECT_REAL_HEIGHT

        #saving updates
        with open(CONFIG_FILE_PATH,"w") as f:
            json.dump(config,f)

        print("config params update sucessfully ")
    
if __name__ == "__main__":
    run()