import os
# import cv2
from PIL import Image
from typing import Union

from fastapi import FastAPI, File, UploadFile
import uvicorn

from TrashClassifier import TrashClassifier

app = FastAPI()

classifier = TrashClassifier()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/classify")
async def classify_waste(img: UploadFile = File(...)):
    img = img.file.read()

    # Create a new file called "pic" in the current directory
    file_path = os.path.join(os.getcwd(), "pic.png")
    with open(file_path, "wb") as f:
        f.write(img)
    try:
         class_img = Image.open(file_path)
         classification = classifier.classify(class_img)
         if classification == "metal":
            return { "classification": "metal"}
         elif classification == "paper":
            return {"classification": "paper"}
         elif classification == "plastic":
            return {"classification": "plastic"}
         elif classification == "glass":
            return {"classification": "glass"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)