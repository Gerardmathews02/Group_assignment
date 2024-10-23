
!apt install tesseract-ocr
!pip install pytesseract
!pip install opencv-python-headless

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

import cv2
import numpy as np
import pytesseract
from google.colab.patches import cv2_imshow

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes


def detect_books(image, net, output_layers, classes):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "book":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i]) for i in indices]

def ocr_books(image, boxes):
    book_titles = []
    for (box, confidence) in boxes:
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)[1]

        text = pytesseract.image_to_string(roi)
        book_titles.append(text.strip())


        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return book_titles

def sort_books(titles):
    return sorted(titles)


def main(image_path):
    net, output_layers, classes = load_yolo()
    image = cv2.imread(image_path)


    detected_books = detect_books(image, net, output_layers, classes)


    book_titles = ocr_books(image, detected_books)

    print("Detected Book Titles:")
    for title in book_titles:
        print(title)


    sorted_titles = sort_books(book_titles)
    print("\nSorted Titles:")
    for title in sorted_titles:
        print(title)

    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
    print(f'Uploaded file: {fn}')
    main(fn)
