from unicodedata import name
import streamlit as st
from PIL import Image
import cv2
import numpy as np

def yolo_detection(img):
    frame_height, frame_width, rgbVal = img.shape
    
    yolo = cv2.dnn.readNet( "D:\\streamlit\\data\\model\\model.cfg","D:\\streamlit\\data\\model\\model.weights")

    classes = []

    with open(".\\data\\model\\obj.names", "r") as f:
        classes = f.read().splitlines()
    
    blob = cv2.dnn.blobFromImage(img, 1/255 , (416,416), [0,0,0], swapRB = True, crop = False)
    yolo.setInput(blob)
    output_layers_name = yolo.getLayerNames()
    layer_output = yolo.forward([output_layers_name[i - 1] for i in yolo.getUnconnectedOutLayers()])

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_output:
        for detection in output:
            score = detection[5:]
            
            class_id = np.argmax(score)
            confidence = score[class_id]

        
            centerX = int(detection[0] * frame_width)
            centerY = int(detection[1] * frame_height)

            width = int(detection[2]* frame_width)
            height = int(detection[3]*frame_height )

            left = int(centerX - width/2)
            top = int(centerY - height/2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])
    indices =  cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.15)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        cv2.rectangle(img, (left, top), (left+width, top + height), (255, 178, 50), 3)

        label = '%.2f' % confidences[i]

        # Get the label for the class name and its confidence
        if classes:
            assert (class_ids[i] < len(classes))
            label = '%s:%s' % (classes[class_ids[i]], label)
        
        #A fancier display of the label from learnopencv.com 
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(img, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                    (255, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (left,top),(left+width, top + height), (255,255,255), 1 )
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    st.text("Predicted Image")    
    st.image(img)
    

def main():
    st.title("Object Detector")
    st.text("This app demonstrates objects detected in images using YOLO-v4")

    # livePred_btn = st.button("Live")
    # if livePred_btn:
    #     liveprediction()

    fromImage_btn = st.button("On Image")
    if fromImage_btn:

        image_file = st.file_uploader("Upload image", type=["jpg","jpeg"])
        
        if image_file is not None:
            org_image = Image.open(image_file)
            st.text("Uploaded image")
            st.image(org_image)
            

            predict_btn  = st.button("Perform Prediction")
            if predict_btn:
                yolo_detection(np.asarray(org_image))
            

if __name__ == "__main__":
    main()