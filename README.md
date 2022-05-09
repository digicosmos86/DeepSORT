# DeepSORT

We run DeepSORT after connecting YOLOv3 and YOLOv3-Tiny object detector by downloading the weights from pjreddy's website. This has been created by training the data on 80 COCO dataset. 
Much of the infrastructure of grabbing pre-trained weights are inspired from YOLOv4-deepsort, which has been adapted for our purposes to take in YOLOv3. https://github.com/theAIGuysCode/yolov4-deepsort
We felt relatively comfortable taking adapatations, as we have implmemented our own version of YOLOv3-Tiny, and were simply benchmarking our performance to the existing deployments of YOLOv3 and YOLOv3-Tiny.

One can obtain respective weights for YOLOv3 and Tiny-YOLOv3 by running the following commands in Terminal:
```
# for YOLOv3 weights
wget https://pjreddie.com/media/files/yolov3.weights

# for YOLOv3-Tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```
Move these `.weight` files into the `data` folder. 


There are ways that we have reviewed that would have allowed for the creation of masks that will count the number of objects.
The ideal version would have been to utilize a webcam, but we soon realized that this proved incredibly difficult as the mask had to be formed prior, and a slight motion in the camera would have immediately altered the mask. Most of this, therefore, happens in the post-processing stage.
Additionally, the training process itself requires some level of pre-training data. 
Hence, we should have been extremely selective in our specification for object counting.
