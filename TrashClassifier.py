import os
import yolov5
import cv2
from ultralytics import YOLO

MODEL_REF= os.path.join(os.getcwd(), "./ai-model/yolov5s.pt")
ONLINE_MODEL_REF = "turhancan97/yolov5-detect-trash-classification"
CUSTOM_MODEL_REF = os.path.join(os.getcwd(), "./ai-model/best.pt")

class TrashClassifier:
    def __init__(self) -> None:
        # load model
        self.model = yolov5.load(MODEL_REF)
        # set model parameters
        self.model.conf = 0.5  # NMS confidence threshold
        self.model.iou = 0.5  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image
        

    def classify(self, img_input):

        # set image
        img = img_input

        # perform inference
        results = self.model(img, size=416)

        # inference with test time augmentation
        results = self.model(img, augment=True)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        if len(scores) == 0:
            return None
        # get the class with the highest score
        highest_score_index = scores.argmax()
        highest_score_class = categories[highest_score_index]

        text = None

        #paper = 2., plastic = 3., metal = 1., glass = 4.

        try:
            if highest_score_class == 1.:
                    text = 'metal'
            elif highest_score_class == 2.:
                    text = 'paper'
            elif highest_score_class == 3.:
                text = 'plastic' 
            elif highest_score_class == 0.:
                text = 'glass'
            img_b = cv2.imread(os.path.join(os.getcwd(), "pic.png"))
            cv2.rectangle(img_b, (int(boxes[highest_score_index][0]), int(boxes[highest_score_index][1])), (int(boxes[highest_score_index][2]), int(boxes[highest_score_index][3])), (0, 255,0), 2)
            cv2.putText(img_b, text, (int(boxes[highest_score_index][0]), int(boxes[highest_score_index][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(os.getcwd(), "result.png"), img_b)
        except Exception as e:
             print(f"Could not write to file: {e}")

        if text:
             print(text)
             return text
        else:
             return None


class TrashClassifierV8:
    def __init__(self) -> None:
        self.model = YOLO(CUSTOM_MODEL_REF)
        self.model.conf = 0.5  # NMS confidence threshold
        self.model.iou = 0.5  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def classify(self, img_input):
        try: 
            prediction = self.model.predict(img_input, show=False, save=False)
            prediction_details = prediction[0].boxes
            class_names = self.model.names
            confidence_scores = prediction_details.conf
            predicted_classes = prediction_details.cls
            x1, y1, x2, y2 = prediction_details.xyxy.tolist()[0]

            if len(confidence_scores) == 0:
                return None
            
            # get the class with the highest score
            highest_score_index = confidence_scores.argmax()
            highest_score_class = predicted_classes[highest_score_index]

            inference = class_names[int(highest_score_class)]
            self.drawBoundinBoxes(x1, y1, x2, y2, inference)

            return inference
        except Exception as e: 
             print(f"Error: {e}")
        
    def drawBoundinBoxes(self, x1, y1, x2, y2, inference):
        img_b = cv2.imread(os.path.join(os.getcwd(), "pic.png"))
        cv2.rectangle(img_b, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255,0), 2)
        cv2.putText(img_b, inference, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(os.getcwd(), "result.png"), img_b)