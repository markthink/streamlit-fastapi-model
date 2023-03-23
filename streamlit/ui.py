import io,requests
import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend = "http://fastapi:8000/segmentation"

def process(image,server_url:str):
    m = MultipartEncoder(fields={"file":("filename",image,"image/jpeg")})
    r = requests.post(server_url,data=m,headers={"Content-Type":m.content_type},timeout=8000)
    return r

st.title("DeepLabV3 image segementation")
st.write("演示通过 fastapi 调用 pytorch 模型完成简单的推理任务!")

input_image = st.file_uploader("请插入一张图片")
if st.button("获取图像分割映射图"):
    col1,col2 = st.columns(2)
    if input_image:
      segments = process(input_image,backend)
      original_image = Image.open(input_image).convert("RGB")
      segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
      col1.header("原始版本")
      col1.image(original_image,use_column_width=True)
      col2.header("分割版本")
      col2.image(segmented_image,use_column_width=True)
    else:
      st.write("请插入一张图片")
