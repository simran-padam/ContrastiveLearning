import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from utils.data_augment import *
from utils.loss_function import *
from utils.encoder_projection import *

def run_Resnet152(train_generator,learning_rate,num_epochs,batch_size,batches_limit):
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    # Create an instance of ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    # Create a generator from the data
    train_generator = train_datagen.flow(train_images, batch_size=batch_size, shuffle=True)
    
    
    simclr_encoder = SimCLREncoder()

    #the paper used LARS, but we use Adam optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

    temperature = 0.5  #temperature parameter for NT-Xent loss
    cumulative_loss = 0.0

    best_loss = float('inf')  

    #training
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs}")
        simclr_train_generator = SimCLRDataGenerator(train_generator)

        augmented_data = simclr_train_generator.generate()

        batch_generator = simclr_encoder.process_batch(augmented_data)

        cumulative_loss = 0.0
        batch_index = 0

        #limit the number of batches to 100 per epoch to save time
        while batch_index < batches_limit:

            #fetch the next batch of augmented images
            with tf.GradientTape() as tape:

                batch_index += 1
                projected_representations_1, projected_representations_2 = next(batch_generator)

                #concatenate the representations along the batch dimension
                concatenated_representations = tf.concat([projected_representations_1, projected_representations_2], axis=0)

                #compute the NT-Xent loss
                loss = nt_xent_loss(concatenated_representations, temperature)

                #compute gradients with respect to the trainable variables of the encoder and projection head
                gradients = tape.gradient(loss, simclr_encoder.encoder_model.trainable_variables + simclr_encoder.projection_head.trainable_variables)

                #apply gradients to update the encoder and projection head
                optimizer.apply_gradients(zip(gradients, simclr_encoder.encoder_model.trainable_variables + simclr_encoder.projection_head.trainable_variables))

                cumulative_loss += loss.numpy()
                print("|", end="")
                if (batch_index % 25) == 0 and (cumulative_loss < best_loss):
                    best_loss = cumulative_loss
                    print(f"batch {batch_index} with improved average loss: {cumulative_loss / batch_index}")

    simclr_encoder.save_models(
    f"resnet152model/encoder_model_lr{learning_rate}_bs{batch_size}_{batches_limit}_{num_epochs}",
    f"resnet152model/projection_head_lr{learning_rate}_bs{batch_size}_{batches_limit}_{num_epochs}"
)

def extract_features(encoder_model, generator,loaded_encoder_model, loaded_projection_head):
    features = []
    labels = []
    steps_per_epoch = generator.n // generator.batch_size

    for imgs, lbls in tqdm(generator, total=steps_per_epoch):
        encoder_features = loaded_encoder_model.predict(imgs)
        batch_features = loaded_projection_head.predict(encoder_features)
        features.append(batch_features)
        labels.append(lbls)
        if len(features) * generator.batch_size >= generator.n:  
            break
    return np.vstack(features), np.vstack(labels)
    
    
def get_accuracy(path_encoder,path_proj):
    loaded_encoder_model, loaded_projection_head = SimCLREncoder.load_models(path_encoder,path_proj)
        
    simclr_encoder = SimCLREncoder(trainable = False)

# Set the loaded models
    simclr_encoder.set_models(loaded_encoder_model, loaded_projection_head)
    
    #load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()



    #convert labels to categorical
    train_labels_categorical = to_categorical(train_labels)
    test_labels_categorical = to_categorical(test_labels)

    train_datagen = ImageDataGenerator(
            rescale=1./255
    )
    
    train_generator = train_datagen.flow(
        train_images,
        train_labels_categorical,
        batch_size=32,
        shuffle=True
    )

    test_datagen = ImageDataGenerator(
            rescale=1./255
    )
    
    test_generator = test_datagen.flow(
        test_images,
        test_labels_categorical,
        batch_size=32,
        shuffle=True
    )
    
    # the above function requires this command
    tf.data.experimental.enable_debug_mode()

    train_features, train_labels = extract_features(simclr_encoder, train_generator,loaded_encoder_model, loaded_projection_head)
    test_features, test_labels = extract_features(simclr_encoder, test_generator,loaded_encoder_model, loaded_projection_head)

    #train a supervised model with 100% labels to check the accuracy
    input_shape = train_features.shape[1:]
    num_classes = 10

    optimizer = Adam(learning_rate=0.001)

    classifier_input = Input(shape=input_shape)

    x = Dense(512, activation='relu')(classifier_input)
    x = Dense(256, activation='relu')(x)

    output = Dense(num_classes, activation='softmax')(x)

    classifier_model = Model(inputs=classifier_input, outputs=output)

    classifier_model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #training
    classifier_model.fit(train_features, train_labels, epochs=50)

    #check test accuracy
    loss, accuracy = classifier_model.evaluate(test_features, test_labels)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    