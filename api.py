"""
API de detecção de peças de xadrez (YOLOv8 + FastAPI).

Como rodar (na pasta do projeto):
    uvicorn api:app --reload

Depois abra http://127.0.0.1:8000/docs para testar pelo navegador.
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent


def encontrar_modelo() -> Path:
    """Ordem: variável MODEL_PATH > models/best.pt > best.pt mais recente em runs/detect/*/weights/."""
    caminho_env = os.getenv("MODEL_PATH")
    if caminho_env:
        return Path(caminho_env)

    modelo_final = BASE_DIR / "models" / "best.pt"
    if modelo_final.exists():
        return modelo_final

    candidatos = sorted(
        BASE_DIR.glob("runs/detect/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidatos:
        raise FileNotFoundError(
            "Nenhum modelo encontrado em runs/detect/*/weights/best.pt. Treine o modelo primeiro."
        )
    return candidatos[0]


CAMINHO_MODELO = encontrar_modelo()
modelo = YOLO(str(CAMINHO_MODELO))  # carregado uma única vez, ao iniciar a API

app = FastAPI(
    title="API de Detecção de Peças de Xadrez",
    description="Recebe uma imagem e retorna, em JSON, as peças de xadrez detectadas pelo modelo YOLOv8.",
    version="1.0.0",
)


@app.get("/")
def status():
    """Verifica se a API está no ar e qual modelo está carregado."""
    return {"status": "online", "modelo": str(CAMINHO_MODELO), "classes": modelo.names}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.5):
    """Recebe uma imagem (UploadFile) e retorna a lista de detecções em JSON."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Envie um arquivo de imagem (JPG, PNG...).")

    conteudo = await file.read()
    try:
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Não foi possível ler o arquivo como imagem.")

    # agnostic_nms=True: caixas sobrepostas de classes diferentes viram uma só (fica a de maior confiança)
    resultado = modelo.predict(source=imagem, conf=conf, agnostic_nms=True, verbose=False)[0]

    deteccoes = []
    for box in resultado.boxes:
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        deteccoes.append(
            {
                "classe": modelo.names[cls_id],
                "classe_id": cls_id,
                "confianca": round(float(box.conf[0].item()), 4),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    return {
        "arquivo": file.filename,
        "largura_imagem": imagem.width,
        "altura_imagem": imagem.height,
        "total_deteccoes": len(deteccoes),
        "deteccoes": deteccoes,
    }
