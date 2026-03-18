from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO("yolov8s.pt")

# Lê a imagem
# img = cv2.imread(r"C:\Users\renan\OneDrive\Desktop\yolo_cachorro\cachorro_carro.jpg")
# img = cv2.imread(r"C:\Users\renan\OneDrive\Desktop\yolo_cachorro\rio_cachorro.png")
img = cv2.imread(r"C:\Users\renan\OneDrive\Desktop\yolo_cachorro\img4.jpg")

if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Faz a detecção
results = model(img)

# Gera a imagem com as caixas e nomes desenhados
img_resultado = results[0].plot()


# cv2.imwrite(r"C:\Users\renan\OneDrive\Desktop\yolo_cachorro\rio_cachorro_resultado.jpg", img_resultado)
cv2.imwrite(r"C:\Users\renan\OneDrive\Desktop\yolo_cachorro\img4_result.jpg", img_resultado)

# Mostra a imagem resultante
cv2.imshow("Resultado YOLO", img_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()