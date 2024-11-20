
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_PATH = r'C:\Users\90506\Downloads\plates\plates' 
cleaned_folder = os.path.join(DATA_PATH, 'train', 'cleaned')
dirty_folder = os.path.join(DATA_PATH, 'train', 'dirty')


def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)
    return image_resized / 255.0


cleaned_images = [f for f in os.listdir(cleaned_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
dirty_images = [f for f in os.listdir(dirty_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

processed_cleaned_images = [preprocess_image(os.path.join(cleaned_folder, img)) for img in cleaned_images]
processed_dirty_images = [preprocess_image(os.path.join(dirty_folder, img)) for img in dirty_images]


X_cleaned = np.array(processed_cleaned_images)
X_dirty = np.array(processed_dirty_images)


y_cleaned = np.zeros(len(X_cleaned))
y_dirty = np.ones(len(X_dirty))


X = np.concatenate((X_cleaned, X_dirty), axis=0)
y = np.concatenate((y_cleaned, y_dirty), axis=0)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)


def create_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_custom_cnn()
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
    verbose=1
)


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()


val_predictions = (model.predict(X_val) > 0.5).astype("int32")


misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_val, val_predictions.flatten())) if true != pred]

for idx in misclassified_indices[:5]:  
    plt.imshow(X_val[idx])
    true_label = "Cleaned (Temiz)" if y_val[idx] == 0 else "Dirty (Kirli)"
    predicted_label = "Cleaned (Temiz)" if val_predictions[idx] == 0 else "Dirty (Kirli)"
    plt.title(f"Gerçek Etiket: {true_label}\nTahmin Edilen: {predicted_label}", fontsize=12)
    plt.axis('off')
    plt.show()


model.save('custom_cnn_plate_classifier.h5')
print("Model başarıyla kaydedildi.")
