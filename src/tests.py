import csv
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

# Número de classes (dígitos de 0 a 9)
num_classes = 10

# Dimensões das imagens
img_rows, img_cols = 28, 28

# Carregar e preparar o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar as imagens para valores entre 0 e 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Ajustar o formato dos dados com base no backend do Keras
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Converter os rótulos para formato one-hot

# Converter as classes em matrizes binárias (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Função para criar o modelo
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])
    return model

# Parâmetros para teste
batch_sizes = [16, 32, 64, 128]  # Diferentes valores para batch_size
epoch_values = [2, 5, 7, 9, 1]   # Diferentes valores para epochs

# Criar arquivo CSV para salvar os resultados
csv_file = 'model_results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Batch Size', 'Epochs', 'Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss'])

# Iterar sobre os parâmetros e treinar/avaliar o modelo
results = []
for batch in batch_sizes:
    for epoch in epoch_values:
        print(f'Training with batch_size={batch}, epochs={epoch}')
        
        # Criar o modelo
        model = create_model()
        
        # Treinar o modelo
        history = model.fit(x_train, y_train, 
                            batch_size=batch, 
                            epochs=epoch, 
                            verbose=1, 
                            validation_data=(x_test, y_test))
        
        # Avaliar os resultados
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # Salvar os resultados
        results.append([batch, epoch, train_acc, val_acc, train_loss, val_loss])
        
        # Escrever no arquivo CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([batch, epoch, train_acc, val_acc, train_loss, val_loss])