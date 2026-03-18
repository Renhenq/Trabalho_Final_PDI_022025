
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
# PRE-PROCESSAMENTO
# ----------------------------

# 1) Suavização
gauss = cv2.GaussianBlur(img, (5, 5), 0)

# 2) Melhoria de contraste com CLAHE
lab = cv2.cvtColor(gauss, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

lab_clahe = cv2.merge((l_clahe, a, b))
contraste = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
preprocessed = contraste

# ----------------------------
# DETECÇÃO
# ----------------------------
output = preprocessed.copy()
# output = img.copy()

results = model.predict(
    source=preprocessed,
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
base = base / "resultados" / "preproc"
pasta = base / nome_imagem
pasta.mkdir(parents=True, exist_ok=True)

# salvar resultados
cv2.imwrite(os.path.join(pasta, "01_original.jpg"), img)
cv2.imwrite(os.path.join(pasta, "02_gauss.jpg"), gauss)
cv2.imwrite(os.path.join(pasta, "03_contraste.jpg"), contraste)
cv2.imwrite(os.path.join(pasta, "04_preprocessed.jpg"), preprocessed)
cv2.imwrite(os.path.join(pasta, "05_resultado.jpg"), output)

print(f"Quantidade de pessoas detectadas: {contador}")

cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", gauss)
cv2.imshow("Contraste CLAHE", contraste)
cv2.imshow("Pre-processada", preprocessed)
cv2.imshow("Deteccao", output)

cv2.waitKey(0)
cv2.destroyAllWindows()