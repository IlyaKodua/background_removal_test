import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.model import BiRefNetModel

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["birefnet"] = BiRefNetModel()
    yield
    ml_models.clear()


app = FastAPI(title="BiRefNet Background Removal", lifespan=lifespan)



@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    result = ml_models["birefnet"].remove_background(image)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
