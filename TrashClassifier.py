import os
import yolov5
import cv2

MODEL_REF= os.path.join(os.getcwd(), "./ai-model/yolov5s.pt")
ONLINE_MODEL_REF = "turhancan97/yolov5-detect-trash-classification"

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



# print(classify(cv2.imread("test.png")))
    # cv2.rectangle(img, (int(boxes[highest_score_index][0]), int(boxes[highest_score_index][1])), (int(boxes[highest_score_index][2]), int(boxes[highest_score_index][3])), (0, 255,0), 2)
    # cv2.imshow("Grey", img)
    # cv2.imwrite("output/result.png", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()