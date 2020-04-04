import cv2
import numpy as np
import time
import dlib

'''
Ref:
https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
'''

#================== Init ========================
detector = dlib.get_frontal_face_detector()


YOLO_MODEL_PATH = "weights/yolov3_final.weights"
YOLO_CONFIG_PATH = "yolo_cfg/yolov3.cfg"
yolo_net = cv2.dnn.readNet(YOLO_CONFIG_PATH, YOLO_MODEL_PATH)
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold


def dlibFaceDetector(np_img):
    #Based on HoG
    # gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    FaceRect,scores,_ = detector.run(np_img,1)
    if FaceRect and scores[0] > 0.888888:
        isfaceFound = True
        for fd in FaceRect:
            x = fd.left()
            y = fd.top()
            w = fd.right() - x
            h = fd.bottom() - y
            np_img = cv2.rectangle(np_img,(x-15,y-25),(x+w+10,y+h+10),(0,0,200),2)
            cv2.putText(np_img,'No Helmet' ,(int(x), int(y+0.01*np_img.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    else:
        isfaceFound =  False
    return np_img,isfaceFound

def applyYoloHelmetDetector(np_img,yolo_net):
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image
    Width = np_img.shape[1]
    Height = np_img.shape[0]
    
    blob = cv2.dnn.blobFromImage(np_img, 1 / 255.0, (inpWidth,inpHeight), swapRB=True, crop=False)

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
            
            if confidence > 0.88:            
                center_x = int(detect[0] * Width)
                center_y = int(detect[1] * Height)
                w = int(detect[2] * Width)
                h = int(detect[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                cv2.rectangle(np_img, (int(x), int(y)), (int(x+w), int(y+h)), (0,200,0), thickness=2)
                cv2.putText(np_img,'With Helmet' ,(int(x), int(y+.05*Height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return np_img

def PreproVideoWrite():
    vid_cap = cv2.VideoCapture('sample_img/meamar_helmet.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # vid_wrt = cv2.VideoWriter('sample_img/meamarprep_helmet.avi', fourcc,40.0,(650,400))
    while(vid_cap.isOpened()):
        ret,frame = vid_cap.read()
        if not ret:
            print('Error Ala re!!!')
            break
        frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame,(650,400),interpolation=cv2.INTER_LANCZOS4)
        frame = dlibFaceDetector(frame)
        # vid_wrt.write(frame)
        cv2.imshow('meamarp',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid_cap.release()
    # vid_wrt.release()
    cv2.destroyAllWindows()
    return
#===============================================================================
if __name__ == "__main__":
    vid_cap = cv2.VideoCapture('sample_img/meamarprep_helmet.avi')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # vidOut = cv2.VideoWriter('sample_img/output.avi',fourcc,30.0,(650,400))
    while(vid_cap.isOpened()):
        ret,frame = vid_cap.read()
        if not ret:
            print('Error Ala re!!!')
            break
        frame,isfaceFound = dlibFaceDetector(frame)
        if not isfaceFound:
            frame = applyYoloHelmetDetector(frame,yolo_net)
        # vidOut.write(frame)
        # cv2.imshow('meamarp',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    vid_cap.release()
    # vidOut.release()
    cv2.destroyAllWindows()


