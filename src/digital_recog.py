from keras.models import load_model
from collections import deque
import numpy as np
import cv2 
from PIL import Image, ImageDraw, ImageFont


extenso= {0: 'zero',
          1: 'um',
          2: 'dois',
          3: 'três',
          4: 'quatro',
          5: 'cinco',
          6: 'seis',
          7: 'sete',
          8: 'oito',
          9: 'nove'
          }
# Carregar o modelo CNN previamente treinado
cnn_model = load_model('mnist_cnn_model.h5')

# Definir limites para a detecção da cor azul no espaço HSV
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Kernel para operações morfológicas
kernel = np.ones((5, 5), np.uint8)

# Inicializar quadro de desenho e outros parâmetros
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
points = deque(maxlen=512)  # Armazena os pontos desenhados
ans = ' '  # Resposta do modelo
rnumb = np.random.randint(8)
score = '000000'

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Caminho da fonte no sistema
font = ImageFont.truetype(font_path, 25)

# Iniciar captura de vídeo
camera = cv2.VideoCapture(0)


# Loop principal
while True:
    # Ler o quadro da câmera
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = cv2.flip(frame, 1)  # Espelhar o quadro horizontalmente
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converter para HSV

    # Criar uma máscara para a cor azul
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    
    # Encontrar contornos na máscara
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Verificar se foram encontrados contornos
    if len(cnts) > 0:
        # Selecionar o maior contorno
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Desenhar círculo em volta do contorno
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        points.appendleft(center)

    elif len(cnts) == 0:
        if len(points) != 0:
            # Pré-processar o quadro desenhado para identificar o dígito
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if blackboard_cnts and len(blackboard_cnts) > 0:
                cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y-10:y+h+10, x-10:x+w+10]

                    if digit.size > 0:
                        newImage = cv2.resize(digit, (28, 28))
                        newImage = np.array(newImage, dtype='float32') / 255.0
                        ans = cnn_model.predict(newImage.reshape(1, 28, 28, 1))[0]
                        ans = np.argmax(ans)
                        
                        if str(ans) == str(rnumb):

                            rnumb = np.random.randint(8)
                            intscore = int(score) + 10
                            score = str(intscore)
                            z = 6 - len(score)
                            for i in range(0, z):
                                score = '0' + score

                    
            # Resetar os pontos e o quadro de desenho
            points = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

    # Conectar os pontos desenhados com linhas
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
        cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

    # Exibir o resultado no quadro
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 410), f"Número identificado: {str(ans)}", font=font, fill=(250, 250, 250, 0))
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Renderizar texto UTF-8 no quadro usando Pillow
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), f"Escreva o número: {extenso[rnumb]}", font=font, fill=(50, 50, 50, 0))
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


    #Pontuação
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((400, 10), f"Pontuação:\n {score}", font=font, fill=(50, 50, 50, 0))
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    
    
    

    # Mostrar o quadro
    cv2.imshow("Digits Recognition Real Time", frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
camera.release()
cv2.destroyAllWindows()
