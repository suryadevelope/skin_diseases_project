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



folderpath = "./dataset/IMG_CLASSES"

userinput = input("What would like to perform T=Train nad Detect or D=detect: ")


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

            self.loss_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.loss_label.pack()
            self.accuracy_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.accuracy_label.pack()
            self.prececision_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.prececision_label.pack()
            self.recall_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.recall_label.pack()
            self.f1_label = tk.Label(root, text="", font=("Helvetica", 16))
            self.f1_label.pack()

            self.image_label = tk.Label(root)
            self.image_label.pack()

        def load_image(self):
            self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
            if self.image_path:
                self.image = Image.open(self.image_path)
                self.image.thumbnail((400, 400))
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
            precision = precision_score(true_labels, predicted_labels, average='weighted')
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
            self.result_label.config(text=class_label)
            self.loss_label.config(text=f"Loss: {loss:.4f}")
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
    detectionmethod()
    
if(userinput == "D"):
    detectionmethod()

elif(userinput == "T"):
    print("train")
    trainmethod()
else:
    sys.exit()


