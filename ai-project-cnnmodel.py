
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, models

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import load_model


images = []
labels = []
for subfolder in tqdm(os.listdir('dataset')):
    subfolder_path = os.path.join('dataset', subfolder)
    for folder in os.listdir(subfolder_path):
        subfolder_path2=os.path.join(subfolder_path,folder)
        for image_filename in os.listdir(subfolder_path2):
            image_path = os.path.join(subfolder_path2, image_filename)
            images.append(image_path)
            labels.append(folder)
df = pd.DataFrame({'image': images, 'label': labels})


plt.figure(figsize=(15,8))
ax = sns.countplot(x='label', data=df, palette='Set1')


ax.set_xlabel("Class",fontsize=20)
ax.set_ylabel("Count",fontsize=20)
plt.title('The Number Of Samples For Each Class',fontsize=20)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()




plt.figure(figsize=(50,50))
for n,i in enumerate(np.random.randint(0,len(df),50)):
    plt.subplot(10,5,n+1)
    img=cv2.imread(df.image[i])
    img=cv2.resize(img,(224,224))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(df.label[i],fontsize=25)



Size = (176, 176)
work_dr = ImageDataGenerator(
    rescale=1./255
    )
train_data_gen = work_dr.flow_from_dataframe(df, x_col='image', y_col='label', target_size=Size, batch_size=6500, shuffle=False)


train_data, train_labels = next(train_data_gen)

X_train, X_test, y_train, y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=5)


X_train[0]

plt.imshow(X_train[0])

cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu',  input_shape=(176, 176, 3)),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(filters=64, kernel_size=(3,3),padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), padding='same'),
    #dense
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
     # Output layer
    layers.Dense(4, activation='softmax')
])
cnn.summary()

tf.keras.utils.plot_model(cnn, to_file='model.png', show_shapes=True, show_layer_names=True,show_dtype=True,dpi=120)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


print("Training DataFrame shape:", train_df.shape)
print("Testing DataFrame shape:", test_df.shape)

test_data_gen = ImageDataGenerator(rescale=1./255)


test_data_gen = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image',
    y_col='label',
    target_size=Size,
    batch_size=1280,  
    shuffle=False
)



x_test, Y_test = next(test_data_gen)


print("X_test shape:", x_test.shape)
print("y_test shape:", Y_test.shape)


unique_classes_train = train_df['label'].unique()


class_dfs_train = np.empty(len(unique_classes_train), dtype=object)


for i, class_label in enumerate(unique_classes_train):
   
    class_df_train = train_df[train_df['label'] == class_label]

    
    class_dfs_train[i] = class_df_train

   
    print(f"Class: {class_label}, Size (Training): {len(class_df_train)}")
    print(class_df_train.head()) 
    print("=" * 50)







from sklearn.utils import resample


max_class_size = max(len(class_df) for class_df in class_dfs_train)


balanced_class_dfs_train = np.empty(len(class_dfs_train), dtype=object)


for i, class_df in enumerate(class_dfs_train):
    
    oversampled_class_df = resample(class_df, replace=True, n_samples=max_class_size, random_state=42)

   
    balanced_class_dfs_train[i] = oversampled_class_df

   
    print(f"Class: {unique_classes_train[i]}, Size (Balanced): {len(oversampled_class_df)}")
    print(oversampled_class_df.head())  
    print("=" * 50)






balanced_train_df = pd.concat(balanced_class_dfs_train, ignore_index=True)


print("Size of Balanced Training DataFrame:", len(balanced_train_df))
print(balanced_train_df.head())





over_data_gen = ImageDataGenerator(rescale=1./255)

# Create test data generator
over_data_gen = over_data_gen.flow_from_dataframe(
    dataframe=balanced_train_df,
    x_col='image',
    y_col='label',
    target_size=Size,
    batch_size=12000,  
    shuffle=True
)





x_train_over, Y_train_over = next(over_data_gen)
# Display the shapes of X_test and y_test
print("x_train_over:", x_train_over.shape)
print("Y_train_over:", Y_train_over.shape)




cnn_over = models.Sequential([
    # CNN layers
    layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(176, 176, 3)),
    layers.MaxPooling2D((2, 2), padding='same'),
    

    layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    

    layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
   

    # Dense layers
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    

    # Output layer
    layers.Dense(4, activation='softmax')
])
cnn_over.summary()





tf.keras.utils.plot_model(cnn_over, to_file='model.png', show_shapes=True, show_layer_names=True,show_dtype=True,dpi=120)


cnn_over.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['accuracy'])



cnn_over.fit(x_train_over, Y_train_over, epochs=10)

cnn_over.save("cnn_model.h5")

cnn_over.evaluate(x_test, Y_test)

predictions = cnn_over.predict(x_test)
y_pred_over = np.argmax(predictions,axis=1)
y_test_over = np.argmax(Y_test,axis=1)
conf_matrix = confusion_matrix(y_test_over, y_pred_over)


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_pred_over), yticklabels=np.unique(y_test_over))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

ClassificationReport = classification_report(y_test_over,y_pred_over)
print('Classification Report is : ', ClassificationReport )





