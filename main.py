# REFERÊNCIAS
#
# Implementação da arquitetura YOLO baseada no artigo YOLO object detection with OpenCV
# do Adrian Rosebrock, autor do site PyImageSearch


# importar os pacotes necessários
import numpy as np
import argparse
import os
import cv2
import time
from imutils.video import VideoStream
from imutils.video import FPS


# constantes do modelo
CONFIDENCE_MIN = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = "yolo-coco"

# receber os argumentos para o script
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Endereço do streaming do drone")
streaming_path = vars(ap.parse_args())['input']

# extrair os nomes das classes a partir do arquivo
print("[+] Carregando labels das classes treinadas...")
with open(os.path.sep.join([MODEL_BASE_PATH, 'coco.names'])) as f:
    labels = f.read().strip().split("\n")

    # gerar cores únicas para cada label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# carregar o modelo treinado YOLO (c/ COCO dataset)
print("[+] Carregando o modelo YOLO treinado no COCO dataset...")
net = cv2.dnn.readNetFromDarknet(
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.cfg']),
    os.path.sep.join([MODEL_BASE_PATH, 'yolov3.weights']))

# extrair layers não conectados da arquitetura YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# iniciar a recepção do streaming
vs = VideoStream(streaming_path).start()
time.sleep(1)
fps = FPS().start()
print("[+] Iniciando a recepção do streaming via RTMP...")

# iterar sobre os frames do streaming
while True:
    frame = vs.read()

    # caso se deseje redimensionar os frames
    # frame = cv2.resize(frame, None, fx=0.2, fy=0.2)

    # capturar a largura e altura do frame
    (H, W) = frame.shape[:2]

    # construir um container blob e fazer uma passagem (forward) na YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # criar listas com boxes, nível de confiança e ids das classes
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filtrar pelo threshold da confiança
            if confidence > CONFIDENCE_MIN and class_id in [0, 1, 2, 3]:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # eliminar ruído e redundâncias aplicando non-maxima suppression
    new_ids = cv2.dnn.NMSBoxes(boxes, confidences,CONFIDENCE_MIN, NMS_THRESHOLD)
    if len(new_ids) > 0:
        for i in new_ids.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # plotar retângulo e texto das classes detectadas no frame atual
            color_picked = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_picked, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_picked, 2)

    # exibir o frame atual
    cv2.imshow('frame', frame)

    # sair, caso seja pressionada a tecla ESC
    c = cv2.waitKey(1)
    if c == 27:
        break

    # atualiza o fps
    fps.update()



# eliminar processos e janelas
fps.stop()
cv2.destroyAllWindows()
vs.stop()
