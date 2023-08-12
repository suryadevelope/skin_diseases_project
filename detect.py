import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.svm import SVC

svmaccuracy = -1
svmclassname = ""
folderpath = "./testDataset"

userinput = input("What would like to perform T=Train and Detect or D=detect: ")


def detectionmethod():

        # Load the trained model
    model = load_model('./skin_disease_model.h5')


    classeslist = sorted(os.listdir(folderpath))
    print(classeslist)

    # Create the Tkinter GUI
    class SkinDiseaseApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Skin Disease Detection")

            self.load_model_button = tk.Button(root, text="Load Image", command=self.load_image)
            self.load_model_button.pack()

            self.predict_button = tk.Button(root, text="Predict", command=self.predict_image, state=tk.DISABLED)
            self.predict_button.pack()

            self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.result_label.pack()

            self.title_label = tk.Label(root, text="CNN model evalution metrics", font=("Helvetica", 16))
            self.title_label.pack()

            self.accuracy_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.accuracy_label.pack()
            self.prececision_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.prececision_label.pack()
            self.recall_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.recall_label.pack()
            self.f1_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.f1_label.pack()

            self.title_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.title_label.pack()
            self.title1_label = tk.Label(root, text="SVM model evalution metrics", font=("Helvetica", 16))
            self.title1_label.pack()

            self.svm_accuracy_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.svm_accuracy_label.pack()
            self.svm_prececision_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.svm_prececision_label.pack()
            self.svm_recall_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.svm_recall_label.pack()
            self.svm_f1_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.svm_f1_label.pack()

            self.image_label = tk.Label(root)
            self.image_label.pack()

        def load_image(self):
            self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
            if self.image_path:
                self.image = Image.open(self.image_path)
                self.image.thumbnail((224, 224))
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.config(image=self.photo)
                self.predict_button.config(state=tk.NORMAL)

        def preprocess_image(self, img_path):
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

        def predict_image(self):
            img = self.preprocess_image(self.image_path)
            svmdetect(self.image_path,self)
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            class_label = "Class " + str(classeslist[class_index])  # Replace with actual class labels
            # Define dataset parameters
           
            img_width, img_height = 224, 224
            batch_size = 32

            # Data preprocessing for testing
           # Data preprocessing for testing
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(
                folderpath,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            # Evaluate the model on the test data
            evaluation = model.evaluate(test_generator)

            # Make predictions on the test data
            predictions = model.predict(test_generator)
            predicted_labels = np.argmax(predictions, axis=1)

            true_labels = test_generator.classes

            # Calculate additional evaluation metrics
            precision = precision_score(true_labels, predicted_labels, average='weighted',zero_division=1)
            recall = recall_score(true_labels, predicted_labels, average='weighted')
            f1 = f1_score(true_labels, predicted_labels, average='weighted')

            # Print evaluation metrics
            loss = evaluation[0]
            accuracy = evaluation[1]

            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1 Score: {f1:.4f}")

            print(svmaccuracy)

            if(svmaccuracy<accuracy*100):
                self.result_label.config(text="Best model is CNN=> Disease = "+class_label)
            else:
                self.result_label.config(text="Best model is SVM=> Disease = "+svmclassname)

            self.accuracy_label.config(text=f"Accuracy: {accuracy*100:.4f} %")
            self.prececision_label.config(text=f"Precision: {precision*100:.4f} %")
            self.recall_label.config(text=f"Recall: {recall*100:.4f} %")
            self.f1_label.config(text=f"f1 score: {f1*100:.4f} %")

    # Create the Tkinter application
    root = tk.Tk()
    app = SkinDiseaseApp(root)
    root.mainloop()

def trainmethod():

    img_width, img_height = 224, 224
    batch_size = 32
    epochs = 10
    num_classes = len(os.listdir(folderpath))

    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        folderpath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Create CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=epochs)

    # Save the model for future use
    model.save('skin_disease_model.h5')
    print("SVM training started")
    trainsvm()

    detectionmethod()

def trainsvm():
    import os
    import numpy as np
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from skimage.io import imread
    from skimage.transform import resize
    import joblib
    from tqdm import tqdm

    def load_image_data(folder_path, image_size):
        data = []
        labels = []
        label_encoder = preprocessing.LabelEncoder()
        
        for label in os.listdir(folder_path):
            label_folder = os.path.join(folder_path, label)
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                image = imread(file_path)
                resized_image = resize(image, image_size)
                data.append(resized_image.flatten())
                labels.append(label)
        
        encoded_labels = label_encoder.fit_transform(labels)
        return np.array(data), encoded_labels, label_encoder


    image_size = (224, 224)

    X, y, label_encoder = load_image_data(folderpath, image_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = svm.SVC(kernel='linear', C=1.0)

    num_iterations = 10  # Number of training iterations

    # Using tqdm for training progress
    with tqdm(total=num_iterations, desc="SVM Training") as pbar:
        for _ in range(num_iterations):
            svm_classifier.fit(X_train, y_train)
            pbar.update(1)

    model_filename = 'svm_image_model.pkl'
    with open(model_filename, 'wb') as model_file:
        joblib.dump((svm_classifier, label_encoder), model_file)

def svmdetect(imgpath,self):
    import os
    import numpy as np
    from skimage.io import imread
    from skimage.transform import resize
    import joblib
    global svmaccuracy,svmclassname

    def load_single_image(image_path, image_size):
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        image = np.expand_dims(img, axis=0)
        flattened_image = image.flatten()
        return flattened_image

    # Load the saved model
    model_filename = 'svm_image_model.pkl'
    with open(model_filename, 'rb') as model_file:
        svm_classifier, label_encoder = joblib.load(model_file)

    # Load a single new image
    image_size = (224, 224)
    X_single_image = load_single_image(imgpath, image_size)

    # Perform prediction on the single image
    predicted_class = svm_classifier.predict([X_single_image])[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    print(f"Predicted Class: {predicted_class}")
    print(f"Predicted Label: {predicted_label}")  

    svmclassname = predicted_label 

        
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

    # Step 1: Set up paths to your dataset
    class_names = sorted(os.listdir(folderpath))
    num_classes = len(class_names)+1

    # Step 2: Load and preprocess images
    image_size = (224, 224)  # You can adjust this based on your needs
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    data_generator = datagen.flow_from_directory(
        folderpath,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    image_count = len(data_generator.filenames)
    X = np.zeros((image_count, *image_size, 3), dtype=np.float32)
    y_true = data_generator.classes

    print(y_true)
    for i, img_path in enumerate(data_generator.filenames):
        img = load_img(os.path.join(folderpath, img_path), target_size=image_size)
        X[i] = img_to_array(img)

    # Step 3: Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    # Step 4: Train an SVM model
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Step 5: Predict and evaluate
    y_pred = svm_model.predict(X_test.reshape(X_test.shape[0], -1))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Manually specify class labels for the classification report
    target_names = class_names  # Use the original class names

    print("Accuracy:", accuracy*100)
    print("Precision:", precision*100)
    print("Recall:", recall*100)
    print("F1-score:", f1*100) 

    svmaccuracy = accuracy*100

    self.svm_accuracy_label.config(text=f"Accuracy: {accuracy*100:.4f} %")
    self.svm_prececision_label.config(text=f"Precision: {precision*100:.4f} %")
    self.svm_recall_label.config(text=f"Recall: {recall*100:.4f} %")
    self.svm_f1_label.config(text=f"f1 score: {f1*100:.4f} %")

if(userinput == "D"):
    detectionmethod()

elif(userinput == "T"):
    print("train")
    trainmethod()
    
else:
    sys.exit()


