import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

class SimCLREncoder:
    def __init__(self, input_shape=(32, 32, 3), trainable=True):
        self.encoder_model = self.create_encoder(input_shape, trainable)
        self.projection_head = self.create_projection_head()

    def create_encoder(self, input_shape, trainable):
        base_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)
        base_model.trainable = trainable
        x = GlobalAveragePooling2D()(base_model.layers[-1].output)
        model = Model(inputs=base_model.input, outputs=x, name='encoder')
        return model

    def create_projection_head(self, input_shape=2048, hidden_units=512, output_units=128):
        projection_head = tf.keras.Sequential([
            Dense(hidden_units, input_shape=(input_shape,), activation='relu'),
            Dense(output_units)
        ])
        return projection_head

    def process_batch(self, generator):
        while True:
            augmented_images_1, augmented_images_2 = next(generator)
            representations_1 = self.encoder_model(augmented_images_1, training=True)
            representations_2 = self.encoder_model(augmented_images_2, training=True)
            projected_representations_1 = self.projection_head(representations_1)
            projected_representations_2 = self.projection_head(representations_2)
            yield projected_representations_1, projected_representations_2
            

    def save_models(self, encoder_path, projection_head_path):
        """Save the encoder and projection head models."""
        self.encoder_model.save(encoder_path)
        self.projection_head.save(projection_head_path)
        print(f"Encoder model saved to {encoder_path}")
        print(f"Projection head saved to {projection_head_path}")

    @staticmethod
    def load_models(encoder_path, projection_head_path):
        """Load and return the encoder and projection head models."""
        encoder_model = tf.keras.models.load_model(encoder_path)
        projection_head = tf.keras.models.load_model(projection_head_path)
        return encoder_model, projection_head
    
    def set_models(self, encoder_model, projection_head):
        """Set the encoder and projection head models."""
        self.encoder_model = encoder_model
        self.projection_head = projection_head