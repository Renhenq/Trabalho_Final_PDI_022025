
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8s.pt")
image_path = "img2.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# ----------------------------
# DETECÇÃO
# ----------------------------
output = img.copy()

results = model.predict(
    source=img,
    conf=0.15,
    imgsz = 640,
    iou=0.45,
    verbose=False
)

contador = 0

for box in results[0].boxes:
    cls = int(box.cls[0].item())
    conf = float(box.conf[0].item())

    if cls == 0:
        contador += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

cv2.putText(
    output,
    f"Total: {contador}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

nome_imagem = os.path.splitext(os.path.basename(image_path))[0]
base = Path(__file__).parent
base = base / "resultados" / "sem_prepoc"
pasta = base / nome_imagem
pasta.mkdir(parents=True, exist_ok=True)

# deteccao utilizando imagem original

cv2.imwrite(os.path.join(pasta, "original.jpg"), img)
cv2.imwrite(os.path.join(pasta, "00_resultado_de_original.jpg"), output)

print(f"Quantidade de pessoas detectadas: {contador}")

cv2.imshow("Original", img)
cv2.imshow("Deteccao", output)

cv2.waitKey(0)
cv2.destroyAllWindows()