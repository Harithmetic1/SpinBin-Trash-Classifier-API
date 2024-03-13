from pydantic import BaseModel

class UploadImageSchema(BaseModel):
    """Models updatable field of a profile instance"""
    img:bytes  