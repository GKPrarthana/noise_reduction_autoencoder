import tensorflow as tf
from pathlib import Path
from src import logger
from src.entity.config_entity import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, datasets: dict):
        self.config = config
        self.datasets = datasets
        self.model = None

    def build_autoencoder(self):
        """Build a convolutional autoencoder for denoising."""
        input_shape = (self.config.image_size, self.config.image_size, 1)  # Grayscale images
        
        # Encoder
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        encoded = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        # Decoder
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        # Autoencoder model
        self.model = tf.keras.Model(inputs, decoded)
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.summary(print_fn=lambda x: logger.info(x))
        
        return self.model

    def train(self):
        """Train the autoencoder using the preprocessed datasets."""
        try:
            logger.info("Starting model training")
            
            # Build the model
            self.build_autoencoder()
            
            # Train the model
            history = self.model.fit(
                self.datasets["train"],
                validation_data=self.datasets["val"],
                epochs=self.config.epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        str(self.config.model_dir / "autoencoder_best.h5"),
                        monitor="val_loss",
                        save_best_only=True
                    )
                ]
            )
            
            # Evaluate on test set
            test_loss = self.model.evaluate(self.datasets["test"])
            logger.info(f"Test loss (MSE): {test_loss}")
            
            # Save the final model
            final_model_path = self.config.model_dir / "autoencoder_final.h5"
            self.model.save(final_model_path)
            logger.info(f"Model saved to {final_model_path}")
            
            logger.info("Model training completed")
            return history
        
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            raise e