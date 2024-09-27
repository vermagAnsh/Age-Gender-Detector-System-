import cv2
import argparse
import os

def detectFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image file or directory of images.', required=True)

args = parser.parse_args()

# Check if the input is provided
if not args.input:
    print("Error: Input path is required.")
    exit()

# Model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "deploy_gender.prototxt"
genderModel = "gender_net.caffemodel"

# Model mean values and lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-8)', '(8-12)', '(13-20)', '(22-32)', '(35-43)', '(46-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load pre-trained models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Check if the input is an image file or a directory of images
input_path = args.input

# Process a single image or all images in a directory
if os.path.isdir(input_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
else:
    image_files = [input_path]

for image_file in image_files:
    frame = cv2.imread(image_file)

    if frame is None:
        print(f"Could not read the image file {image_file}")
        continue

    resultImg, faceBoxes = detectFace(faceNet, frame)
    
    if not faceBoxes:
        cv2.putText(resultImg, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    for faceBox in faceBoxes:
        padding = 20
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Predict gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("Age and Gender Detection", resultImg)

    output_file = f"output_{os.path.basename(image_file)}"
    cv2.imwrite(output_file, resultImg)
    print(f"Saved result to {output_file}")
    cv2.waitKey(0)

cv2.destroyAllWindows()
