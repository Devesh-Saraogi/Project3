from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import base64


#schema
class ImageMetadata(BaseModel):
    category: str = Field(description="Primary category of the product")
    subcategory: Optional[str] = Field(description="Subcategory if applicable")
    colors: list[str] = Field(description="Dominant colors visible in the image")
    material: Optional[str] = Field(description="Material of the product")
    gender: Optional[str] = Field(description="Target gender if applicable")
    style: Optional[str] = Field(description="Style like casual, formal, sporty")
    occasion : Optional[str] = Field(description="Most Probable Occasion to Wear the Product")
    description: str = Field(description="One sentence product description")




model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

image_bytes = open("image1.jpg", "rb").read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
mime_type = "image/jpeg"

msg =  HumanMessage(
        content=[
        {"type": "text", "text": "Analyze the image and extract structured product metadata."},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": mime_type,
        },
    ]
    )

def extract_metadata(model,msg) -> ImageMetadata:
    structured_model = model.with_structured_output(ImageMetadata)
    result = structured_model.invoke([msg])
    return result

metadata = extract_metadata(model,msg)

print(metadata)
print(metadata.model_dump())  # clean JSON