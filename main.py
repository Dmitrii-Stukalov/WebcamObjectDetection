import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open(r'coco.names') as f:
    classes = np.array(f.read().splitlines())

cam = cv2.VideoCapture(0)
key = None
images = []

while key != 27:
    image = cam.read()[1]
    height = image.shape[0]
    width = image.shape[1]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(416, 416))

    net.setInput(blob)
    layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(layers_names)

    boxes = []
    confidences = []
    classes_id = []

    for output in layer_outputs:
        for detected in output:
            scores = detected[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detected[0] * width)
                center_y = int(detected[1] * height)
                w = int(detected[2] * width)
                h = int(detected[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classes_id.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(boxes) != 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[classes_id[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label + ' ' + confidence, (x + w, y + h), font, 2, color, 2)

    cv2.imshow('Image', image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
