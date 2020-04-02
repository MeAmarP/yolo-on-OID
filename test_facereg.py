import face_recognition
import matplotlib.pyplot as plt
import time


start = time.time()
image = face_recognition.load_image_file("sample_img/india_road.jpeg")
face_locations = face_recognition.face_locations(image)
# top,right,bottom,left = face_locations[0]
# h=bottom-top
# w=right-left
# print(bottom,right,top,left,h,w)
print(time.time()-start)
print(face_locations)
# plt.imshow(image[top:top+w,left:left+h])