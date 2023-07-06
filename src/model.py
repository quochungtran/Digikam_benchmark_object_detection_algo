import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
import os 


# Load names of classes and get random colors
classes = open(os.path.join('../yolo_settings/coco.names')).read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


class objectDetectorYolo:
    def __init__(self, yolo_cfg, 
                 yolo_weight, 
                 subClass, 
                 conf_threshold = 0.5,
                 nms_threshold = 0.4,
                 scaleFactor=0.00392, 
                 inputImageSize=(416, 416), 
                 meanValToSubtract = (0,0,0)):
        
        # yolo conf
        self.yolo_cfg       = yolo_cfg
        self.yolo_weight    = yolo_weight

        # params of input blob
        self.subClass       = subClass
        self.scaleFactor    = scaleFactor
        self.inputImageSize = inputImageSize
        self.meanValToSubtract = meanValToSubtract

        # params conf of post_processing
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold

        self.net = self.load_models()
    
    def load_models(self):
        net = cv.dnn.readNetFromDarknet(str(self.yolo_cfg), 
                                              str(self.yolo_weight))

        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)        
        return net

    def get_ouputs_name(self):
        layer_names = self.net.getLayerNames()
        try:
            ouput_layers = [layer_names[i-1]    for i in self.net.getUnconnectedOutLayers()]
        except:
            ouput_layers = [layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        
        return ouput_layers
    
    def post_processing(self, inputImage, outputs):
        H, W = inputImage.shape[:2]

        boxes       = []
        confidences = []
        classIDs    = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > self.conf_threshold:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        print(type(indices))

        user_indices = np.array([i for i in indices if classes[classIDs[i]] in self.subClass])


        if len(user_indices) > 0:
            for i in user_indices:

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]# Load names of classes and get random colors
                cv.rectangle(inputImage, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv.putText(inputImage, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    def detect_objects(self, inputImage):
        
        # generate blobs from input image
        inputBlob = cv.dnn.blobFromImage(inputImage,  self.scaleFactor, self.inputImageSize, swapRB=True, crop=False)
        self.net.setInput(inputBlob)

        # set blob into 
        t0 = time.time()
        outputs = self.net.forward(self.get_ouputs_name())
        t = time.time() - t0
        print(t)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        self.post_processing(inputImage, outputs)

        plt.imshow(cv.cvtColor(inputImage, cv.COLOR_BGR2RGB))
