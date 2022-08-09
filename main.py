from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier
def data_load():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    # Load training data, labels; and testing data and their true labels
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    #Hot encode labels
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Just checking
    print("Train images size:", train_images.shape)
    print("Train labels size:", train_labels.shape)
    print("Test images size:", test_images.shape)
    print("Test label size:", test_labels.shape)
    return(train_images,test_images,train_labels,test_labels,class_names)

def data_load():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    # Load training data, labels; and testing data and their true labels
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    #Hot encode labels
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Just checking
    print("Train images size:", train_images.shape)
    print("Train labels size:", train_labels.shape)
    print("Test images size:", test_images.shape)
    print("Test label size:", test_labels.shape)
    return(train_images,test_images,train_labels,test_labels,class_names)
X_train, X_test, y_train, y_test, class_names = load_data()

# define the search
search = StructuredDataClassifier(max_trials=15)
# perform the search
search.fit(x=X_train, y=y_train, verbose=0)
# evaluate the model
loss, acc = search.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)
# use the model to make a prediction
# row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
# X_new = asarray([row]).astype('float32')
# yhat = search.predict(X_new)
# print('Predicted: %.3f' % yhat[0])
# # get the best performing model
# model = search.export_model()
# # summarize the loaded model
# model.summary()
# # save the best performing model to file
# model.save('model_sdef data_load():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    # Load training data, labels; and testing data and their true labels
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    #Hot encode labels
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Just checking
    print("Train images size:", train_images.shape)
    print("Train labels size:", train_labels.shape)
    print("Test images size:", test_images.shape)
    print("Test label size:", test_labels.shape)
    return(train_images,test_images,train_labels,test_labels,class_names)onar.h5')