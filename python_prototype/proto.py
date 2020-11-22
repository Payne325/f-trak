import cv2
import numpy as np

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

prototxt = "resources/deploy.prototxt.txt"
model = "resources/model.caffemodel" 
min_confidence = 0.9

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

camera = cv2.VideoCapture(0)

window_name = "f-trak_python_prototype"

cv2.namedWindow(window_name)

while True:
   success, frame = camera.read()

   if not success:
      print("No image read from camera!")

   frame = resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
   (h, w) = frame.shape[:2]
   blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and predictions
   net.setInput(blob)
   detections = net.forward()

   # loop over the detections
   for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the prediction
      confidence = detections[0, 0, i, 2]
      
      # filter out weak detections by ensuring the `confidence` is
      # greater than the minimum confidence
      if confidence < min_confidence:
      	continue
      
      # compute the (x, y)-coordinates of the bounding box for the
      # object
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")      
      print((startX, startY, endX, endY))

      # draw the bounding box of the face along with the associated
      # probability
      text = "{:.2f}%".format(confidence * 100)
      
      y = startY - 10 if startY - 10 > 10 else startY + 10
      
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
      
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

   cv2.imshow(window_name, frame)

   k = cv2.waitKey(1)
   if k%256 == 27:
      print("Escape hit, closing...")
      break


camera.release()
cv2.destroyAllWindows()