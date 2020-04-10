# yolo-on-OpenImageDatabse for building Helmet Detector Use Case

Goal Here is to extract BBOX annoted data from Open Image Database for particular class and build object detector using YOLO.
*Selected Class = "Helmet"*

### Approch:
+ Download Img data for selected class
+ Downaload necessary csv files
  + Class Labelmap csv file: holds **ClassID and ClassName**
  + train imgid file: **Image Id and Access URL**
  + train imgage annoted bbox file: **Image ID, Class ID and BBoxe(xmin,xmax,ymin,ymax)** etc.
  + test imgid file **TEST Image Id and Access URL**
  + test image annoted bbox file **TEST Image ID, Class ID and BBoxe(xmin,xmax,ymin,ymax)** etc.

### To Build Helmet Detector
+ As we are only concerned about class "Helmet", Keep classid related records in Annotation files
+ Other option is to keep records for which we have image available in dir.
  + Use the `isin` method `df[df['ImageID'].isin(list_img_id_in_dir)]`
+ Next, we need to convert the OID bbox format to YOLO bbox format.
+ Save `.txt` file for each image id containing bbox details in YOLO format.
+ Put all images and associated bbox.txt files in a single directory.
+ do the above task for train and test dataset.
+ Follow below dir structure.

        --- dataset name
            --- images
                --- img1.jpg
                --- img2.jpg
                ..........
            --- labels
                --- img1.txt
                --- img2.txt
                ..........
            --- train.txt
            --- val.txt

### Thanks:
+ https://storage.googleapis.com/openimages/web/download.html
+ https://github.com/harshilpatel312/open-images-downloader
+ https://stackoverflow.com/a/5612  1386/9018788
+ https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
+ https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/

Setting Up YOLO on Google Colab:
+ **https://github.com/kriyeng/yolo-on-colab-notebook/blob/master/Tutorial_DarknetToColab.ipynb**