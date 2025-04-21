#update components
import tensorflow as tf
from pathlib import Path
from src import logger
from src.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """Initialize with configuration."""
        self.config = config

    def get_base_model(self):
        """Create the base autoencoder model."""
        try:
            self.model = self._build_autoencoder()
            self.save_model(path=self.config.base_model_path, model=self.model)
            logger.info("Base autoencoder model created and saved")
        except Exception as e:
            logger.error(f"Failed to create base model: {e}")
            raise e

    def _build_autoencoder(self):
        """Build the convolutional autoencoder architecture."""
        input_img = tf.keras.Input(shape=(self.config.params["IMAGE_SIZE"][0],
                                       self.config.params["IMAGE_SIZE"][1], 1))
        # Encoder
        x = tf.keras.layers.Conv2D(self.config.params["MODEL"]["FILTERS"][0],
                                 self.config.params["MODEL"]["KERNEL_SIZE"],
                                 activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(self.config.params["MODEL"]["FILTERS"][1],
                                 self.config.params["MODEL"]["KERNEL_SIZE"],
                                 activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        # Decoder
        x = tf.keras.layers.Conv2D(self.config.params["MODEL"]["FILTERS"][1],
                                 self.config.params["MODEL"]["KERNEL_SIZE"],
                                 activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(self.config.params["MODEL"]["FILTERS"][0],
                                 self.config.params["MODEL"]["KERNEL_SIZE"],
                                 activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(1, self.config.params["MODEL"]["KERNEL_SIZE"],
                                      activation='sigmoid', padding='same')(x)
        autoencoder = tf.keras.Model(input_img, decoded)
        return autoencoder

    @staticmethod
    def _prepare_full_model(model, learning_rate):
        """Compile the autoencoder model."""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mae"]  # Mean Absolute Error for monitoring
        )
        model.summary(print_fn=lambda x: logger.info(x.encode('ascii', 'ignore').decode('ascii')))

        return model

    def update_base_model(self):
        """Prepare and save the full autoencoder model."""
        try:
            self.full_model = self._prepare_full_model(
                model=self.model,
                learning_rate=self.config.params["MODEL"]["LEARNING_RATE"]
            )
            self.save_model(path=self.config.base_model_path, model=self.full_model)
            logger.info("Full autoencoder model updated and saved")
        except Exception as e:
            logger.error(f"Failed to update base model: {e}")
            raise e

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the model to the specified path."""
        try:
            model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise e