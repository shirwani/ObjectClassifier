import copy
import numpy as np
import h5py
import pickle

def load_dataset():
    train_dataset = h5py.File('datasets/train_cats.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_cats.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    print(train_set_x_orig.shape)
    print(train_set_y_orig.shape)
    print(classes)

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim,1), dtype=float)
    b = 0.
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * (np.sum(np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)))
    dw = (1/m) * np.dot(X,(A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


class LogisticRegressionModel:
    def __init__(self):
        self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y, self.classes = load_dataset()
        self.m_train = self.train_set_x_orig.shape[0]
        self.m_test = self.test_set_x_orig.shape[0]
        self.num_px = self.train_set_x_orig.shape[1]
        self.train_set_x_flatten = self.train_set_x_orig.reshape(self.train_set_x_orig.shape[0], -1).T
        self.test_set_x_flatten = self.test_set_x_orig.reshape(self.test_set_x_orig.shape[0], -1).T
        self.train_set_x = self.train_set_x_flatten / 255.
        self.test_set_x = self.test_set_x_flatten / 255.
        self.model = None

        print(self.train_set_x_flatten.shape)


    def train_model(self):
        print("Generating model...")
        self.model = model(self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y,
                           num_iterations=2000, learning_rate=0.005, print_cost=True)
        print("Model is ready!")

    def get_model(self):
        return self.model

    def get_num_px(self):
        return self.num_px

    def get_classes(self):
        return self.classes

if __name__ == '__main__':
    training = LogisticRegressionModel()
    training.train_model()
    data = {
        "model": training.get_model(),
        "num_px": training.get_num_px(),
        "classes": training.get_classes()
    }
    with open("model.pkl", 'wb') as file:
        pickle.dump(data, file)



