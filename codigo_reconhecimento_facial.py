import dlib
import cv2
import os,sys
import numpy as np
import tensorflow as tf
from typing import Union
import time
import tkinter as tk
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk

model_path = "mobilefacenet.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Obter detalhes dos tensores de entrada e saida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_detector = dlib.get_frontal_face_detector()
file_path = 'encodings_tflite.txt'
embeddings_to_save =[]

def save_data_single(file_path, data):
    with open(file_path, 'a') as file:
            embedding_str = ' '.join(map(str, data[0]))
            file.write(f"{embedding_str} {data[1]}\n")

def load_data(file_path):
    embeddings_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            embedding = np.array([float(value) for value in values[:256]])
            name = ' '.join(values[256:])
            embeddings_list.append((embedding, name))

    return embeddings_list

def findCosineDistance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_face_locations(imagem):
    
    image_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    face_coord = face_detector(image_gray, 1)    
    return face_coord

def extract_face2(image , face_coord, resolution=(112, 112),):
    if len(face_coord) > 0:
        (x, y, w, h) = rect_to_bb(face_coord[0])
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        # cortar rosto
        image_face = image[y1:y2, x1:x2, :]

        image_face = np.asarray(image_face, dtype="float32")
        if image_face.shape[0] != 0 and image_face.shape[1] and image_face.shape[2] != 0:

            image_resized = cv2.resize(image_face, resolution, interpolation = cv2.INTER_AREA)
            image_resized /= 255
            image_resized = np.expand_dims(image_resized, axis=0)
            
            return image_resized
    else:
        return None

def get_embedding(image):
    # Alimentar a imagem para o modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    # Obter os resultados (embedding)
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    # Converter o embedding para um array NumPy
    embedding = np.array(embedding)

    return embedding

class FaceRecognition:
    face_locations = [] 
    face_encodings = []
    face_names = []
    tupla_encodings_nome = []
    process_current_frame = True
    def __init__(self):
        self.tupla_encodings_nome = load_data(file_path)

    def adicionar_nome(self,image, encoding):
            # Dialogo para inserir o nome da pessoa
            nome = askstring("Adicionar Nome", "Digite o seu Nome e Sobrenome:")
            while nome == '':
                nome = askstring("Adicionar Nome", "Nome invalido. Digite o seu Nome e Sobrenome:")
            
            # Se o usuario pressionar cancelar, o nome sera None
            if (nome is not None and encoding is not None):
                original_nome = nome
                index = 1

                while any(existing_nome == nome for _, existing_nome in self.tupla_encodings_nome):
                    nome = f"{original_nome} {index}"
                    index += 1
        
                save_data_single(file_path,(encoding,nome))
                cv2.imwrite(f'faces/{nome}.png', image)
                self.tupla_encodings_nome.append((encoding,nome))
        
    def run_recognition(self):
        # pega a primeira camera disponivel
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if os.environ.get('DISPLAY','') == '':
            print('no display found. Using:0.0')
            os.environ.__setitem__('DISPLAY', ':0.0')
        else:
            print(os.environ.get('DISPLAY',''))
        # Inicializar a interface grafica
        root = tk.Tk()

        root.attributes('-fullscreen', True)
        root.bind('<f>', lambda event: root.attributes('-fullscreen', not root.attributes('-fullscreen')))

        screen_width = root.winfo_screenwidth()

        # Criar um rotulo para exibir a imagem
        label = tk.Label(root)
        
        #trocar 1280 e 720 pela resolucao da imagem pega pela webcam.
        label.grid(row=0, column=0, padx=(screen_width - 1280) // 2)

        # Criar um botão na interface grafica
        botao = tk.Button(root, text="Adicionar Seu Rosto", height=2, width=18, bg="red", fg="yellow")
        botao.config(font=("Arial", 14))
        botao.grid(row=1, column=0, pady=10)

        root.bind('<Return>', lambda event=None: self.adicionar_nome(small_frame, self.face_encodings[0]))
        botao["command"] = lambda: self.adicionar_nome(small_frame, self.face_encodings[0])

        if not video_capture.isOpened():
            sys.exit('camera nao encontrada')
            
        while True:
            ret, frame = video_capture.read()
            
            #Processa 1 vez a cada 2 frames
            if (self.process_current_frame):
                small_frame = cv2.resize(frame, (0,0), fx=1, fy=1)
                self.face_locations = get_face_locations(small_frame)
                
                face_image = extract_face2(small_frame, self.face_locations)
                self.face_encodings = []
                
                if face_image is not None:
                    self.face_encodings.append(
                        get_embedding(face_image))
                    
                self.face_names = []
                for face_encoding in self.face_encodings:
                    name = 'Desconhecido'
                    #array com as distancias de cada rosto conhecido com o do frame atual
                    face_distances = [findCosineDistance(known_encoding, face_encoding) for known_encoding, _ in self.tupla_encodings_nome]
                    #determina o elemento do array com menor distancia
                    best_match_index = np.argmin(face_distances)

                    #verifica se o elemento e menor que o threshold do modelo
                    if face_distances[best_match_index] < 0.30:
                        name = self.tupla_encodings_nome[
                            best_match_index][1]
                    self.face_names.append(f'{name}')   
            self.process_current_frame = not self.process_current_frame
            
            #Mostrar anotacoes na imagem
            if len(self.face_locations) > 0:
                for face_location, name in zip(self.face_locations, self.face_names):
                    top = face_location.top()
                    right = face_location.right()
                    bottom = face_location.bottom()
                    left = face_location.left()
                    
                    if name != 'Desconhecido':
                        text = 'Bem vindo(a), ' + name + '!'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = 50  
                        
                        # Define a cor amarela (BGR)
                        color = (0, 255, 255)
                        font_scale = 1.2
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                    else:
                        text = 'Ola! Voce ainda nao esta cadastrado'
                        text2 = 'Clique no botao abaixo ou aperte Enter'
                        text3 = 'Quando o seu rosto estiver enquadrado'
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]

                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = 50  
                        color = (0, 255, 255)
                        
                        cv2.putText(frame, text, (text_x+60, text_y), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                        cv2.putText(frame, text2, (text_x+60, text_y+50), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                        cv2.putText(frame, text3, (text_x+60, text_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

                    cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)

            frame_stretched = cv2.resize(frame, (1280, 720))

            frame_rgb = cv2.cvtColor(frame_stretched, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            label.configure(image=img_tk)
            root.update()
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
                
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()