import cv2
import numpy as np
import time
import dlib


"""
#TODO:
1. On Given frame run face detector
1.1 Add a way to adjust brightness and contrast
2. If face detected tagged them as Helmet Violaters
2.1 Apply non-max suppression
3. Show RED Colored BBox for Rule Violaters
4. Show GREEN Colored BBox for Keepers
5. Add Flag if want to display output result
"""

#================== Init ========================
detector = dlib.get_frontal_face_detector()


YOLO_MODEL_PATH = "weights/yolov3_final.weights"
YOLO_CONFIG_PATH = "yolo_cfg/yolov3.cfg"
net = cv2.dnn.readNet(YOLO_CONFIG_PATH, YOLO_MODEL_PATH)
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

def dlibFaceDetector(np_img):
    # gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    FaceRect = detector(np_img,1)
    for fd in FaceRect:        
        x = fd.rect.left()
        y = fd.rect.top()
        w = fd.rect.right() - x
        h = fd.rect.bottom() - y
        np_img = cv2.rectangle(np_img,(x-15,y-25),(x+w+10,y+h+10),(0,0,200),2)
    return np_img

def applyYoloHelmetDetector(np_img,yolo_net):
    # Initialize the parameters
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (inpWidth,inpHeight), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names =  yolo_net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
    objdetects = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in objdetects:
        for detect in out:
            scores=detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:            
                # print(confidence)
                center_x = int(detect[0] * Width)
                center_y = int(detect[1] * Height)
                w = int(detect[2] * Width)
                h = int(detect[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,200,0), thickness=2)
                cv2.putText(img,'With Helmet' ,(int(x), int(y+.05*Height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return img

#==============================================================================


IMG_PATH = "sample_img/indiaroad1.jpeg"
YOLO_MODEL_PATH = "weights/yolov3_final.weights"
YOLO_CONFIG_PATH = "yolo_cfg/yolov3.cfg"

start = time.time()
img = cv2.imread(IMG_PATH)
Width = img.shape[1]
Height = img.shape[0]


# img = applyHaarFaceDetector(img)
img = dlibFaceDetector(img)
# img = applyYoloHelmetDetector(img,net)
print("Opertion Times:",time.time()-start)


#=========================================HaarCascade - Face Detect============
# PATH_TO_XML = '/home/amarp/Documents/pyproj/CV/forFoilio/yolo-on-OID/HaarTrained/frontalFace10/haarcascade_frontalface_alt.xml'
# face_cascade = cv2.CascadeClassifier(PATH_TO_XML)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# # gray = clahe.apply(gray)
# faces = face_cascade.detectMultiScale(gray)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x-15,y-25),(x+w+10,y+h+10),(255,0,0),2)

#=======YOLO - Helmet Detect
# net = cv2.dnn.readNet(YOLO_CONFIG_PATH, YOLO_MODEL_PATH)
# blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# net.setInput(blob)

# layer_names =  net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# objdetects = net.forward(output_layers)
# print(type(objdetects))

# class_ids = []
# confidences = []
# boxes = []

# for out in objdetects:
#     for detect in out:
#         scores=detect[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
        
#         if confidence > 0.5:            
#             print(confidence)
#             center_x = int(detect[0] * Width)
#             center_y = int(detect[1] * Height)
#             w = int(detect[2] * Width)
#             h = int(detect[3] * Height)
#             x = center_x - w / 2
#             y = center_y - h / 2
#             class_ids.append(class_id)
#             confidences.append(float(confidence))
#             boxes.append([x, y, w, h])
#             cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,200,0), thickness=2)
#             cv2.putText(img,obj_class[0] ,(int(x), int(y+.05*Height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()