import io
from segmentation import get_segmentator,get_segments
from starlette.responses import Response

from fastapi import FastAPI,File

model = get_segmentator()

app = FastAPI(
    title='DeepLabV3 image segmentation',
    description='演示如何使用 fastapi快速部署 pytorch模型',
    version='0.1.0'
)
@app.post('/segmentation')
def get_segmentation(file:bytes=File(...)):
    segmented_image = get_segments(model,file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io,format='PNG')
    return Response(bytes_io.getvalue(),media_type='image/png')