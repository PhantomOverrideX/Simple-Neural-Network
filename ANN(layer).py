import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, keepdims=True)

input_size = 784
output_size = 10
hidden_layer_size = 64
hidden_layers = 3
lr = 0.01

try:
    data = np.load("model.npz")

    w1 = data["w1"]
    b1 = data["b1"]

    w2 = data["w2"]
    b2 = data["b2"]

    w3 = data["w3"]
    b3 = data["b3"]
    
    w4 = data["w4"]
    b4 = data["b4"]

    w5 = data["w5"]
    b5 = data["b5"]

except FileNotFoundError:

    w1 = np.random.randn(hidden_layer_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros(hidden_layer_size)

    w2 = np.random.randn(hidden_layer_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b2 = np.zeros(hidden_layer_size)
    
    w3 = np.random.randn(hidden_layer_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b3 = np.zeros(hidden_layer_size)
    
    w4 = np.random.randn(hidden_layer_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b4 = np.zeros(hidden_layer_size)

    w5 = np.random.randn(output_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b5 = np.zeros(output_size)

weights = [w1, w2, w3, w4, w5]
biases = [b1, b2, b3, b4, b5]

def input_layer(weight, image, bias):
    z = weight @ image + bias
    return(sigmoid(z), z)

def hidden_layer(a, bias, weight):
    z = weight @ a + bias
    return(sigmoid(z), z)

def output_layer(a, bias, weight):
    z = weight @ a + bias
    return(softmax(z))

def loss_calc(label, prediction):
    y = np.zeros(10)
    y[label] = 1

    loss = -np.sum(y * np.log(prediction + 1e-9))
    return(loss, y)

def begin_backprop(prediction, y, a4):
    dz5 = prediction - y
    
    dw5 = np.outer(dz5, a4)
    db5 = dz5
    return(db5, dz5, dw5) 
 
def backprop(dz5, weight, a3, z4):
    da4 = weight.T @ dz5
    dz4 = da4 * sigmoid_deriv(z4)

    dw4 = np.outer(dz4, a3)
    db4 = dz4
    return(db4, dz4, dw4)

def Gradient_clipping_1(dw5, db5):
    dw4 = np.clip(dw5, -1, 1)
    db4 = np.clip(db5, -1, 1)
    return(dw4, db4)

def Gradient_clipping_2(dw5, db5, i):
    global weights, biases
    weights[i] -= lr * dw5
    biases[i] -= lr * db5

def Save_model():
    w1 = weights[0]
    b1 = biases[0]
    w2 = weights[1]
    b2 = biases[1]
    w3 = weights[2]
    b3 = biases[2]
    w4 = weights[3]
    b4 = biases[3]
    w5 = weights[4]
    b5 = biases[4]
    
    np.savez(
            "model.npz",
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w3=w3,
            b3=b3,
            w4=w4,
            b4=b4,
            w5=w5,
            b5=b5
        )

def Train_layers():
    print("Train")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    global weights, biases, input_size, hidden_layers
    
    eproch_nmu = int(input("eproch: "))
    
    for eproch in range(eproch_nmu):

        indices = np.random.permutation(60000)

        correct = 0    

        for j in indices:

            image = train_images[j] / 255.0
            image = image.reshape(input_size)

            label = train_labels[j]

            # Forward pass: store (activation, z) for every layer, including the input
            back_prop = [(image, None)]

            prediction, z = input_layer(weights[0], image, biases[0])
            back_prop.append((prediction, z))

            for i in range(hidden_layers):
                i += 1
                prediction, z = hidden_layer(prediction, biases[i], weights[i])
                back_prop.append((prediction, z))

            prediction = output_layer(prediction, biases[4], weights[4])

            loss, y = loss_calc(label, prediction)

            # Backprop: build gradients for all 5 weight matrices
            dbdzdw = [None] * 5

            db5, dz5, dw5 = begin_backprop(prediction, y, back_prop[4][0])
            dbdzdw[4] = (db5, dz5, dw5)

            dz = dz5
            for k in range(hidden_layers + 1):
                db, dz_new, dw = backprop(dz, weights[4 - k], back_prop[3 - k][0], back_prop[4 - k][1])
                dbdzdw[3 - k] = (db, dz_new, dw)
                dz = dz_new

            dwdb = []
            for i in range(hidden_layers + 2):
                dw4, db4 = Gradient_clipping_1(dbdzdw[i][2], dbdzdw[i][0])
                dwdb.append((dw4, db4))

            for i in range(hidden_layers + 2):
                Gradient_clipping_2(dwdb[i][0], dwdb[i][1], i)

            if np.argmax(prediction) == label:
                correct += 1

            if j % 10000 == 0:
                print("Step:", j)
                print("Loss:", loss)
                print("Guess:", np.argmax(prediction))
                print("Label:", label)
                print()

        print("eproch:", eproch)
        print("Accuracy:", (correct / 60000) * 100)
        print()
        
        Save_model()

def Test_layers():
    print("Test")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    global weights, biases, input_size, hidden_layers

    correct = 0

    for j in range(10000):

        image = test_images[j] / 255.0
        image = image.reshape(784)

        label = test_labels[j]

        prediction, z = input_layer(weights[0], image, biases[0])

        for i in range(hidden_layers):
            i += 1
            prediction, z = hidden_layer(prediction, biases[i], weights[i])

        prediction = output_layer(prediction, biases[4], weights[4])

        loss, y = loss_calc(label, prediction)

        if np.argmax(prediction) == label:
            correct += 1

        print("Step:", j)
        print("Loss:", loss)
        print("Guess:", np.argmax(prediction))
        print("Label:", label)
        print()

    print("Accuracy:", (correct / 10000) * 100)
    print()

    return (correct / 10000) * 100

def user_input():
    user_in = input("Train or Test: ")

    if user_in.lower() == "train":
        Train_layers()

    elif user_in.lower() == "test":
        Test_layers()

    else:
        exit()

user_input()
input("Press Enter to Exit")