#update components
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src import logger
from src.entity.config_entity import EvaluationConfig
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from src.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Evaluate the model on the test set and compute metrics."""
        self.model = self.load_model(self.config.path_of_model)
        
        # Evaluate on test set (compute test MSE)
        self.score = self.model.evaluate(self.config.test_data)
        logger.info(f"Test loss (MSE): {self.score}")
        
        # Get a batch from the test set for visualization and additional metrics
        test_batch = next(iter(self.config.test_data.take(1)))
        noisy_images, clean_images = test_batch[0], test_batch[1]
        
        # Predict denoised images
        predicted_images = self.model.predict(noisy_images)
        
        # Compute PSNR and SSIM
        psnr_values = []
        ssim_values = []
        for i in range(len(clean_images)):
            clean_img = clean_images[i].numpy().squeeze()
            pred_img = predicted_images[i].squeeze()
            psnr_val = psnr(clean_img, pred_img, data_range=1.0)
            ssim_val = ssim(clean_img, pred_img, data_range=1.0)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
        
        self.avg_psnr = np.mean(psnr_values)
        self.avg_ssim = np.mean(ssim_values)
        logger.info(f"Average PSNR: {self.avg_psnr:.2f} dB")
        logger.info(f"Average SSIM: {self.avg_ssim:.4f}")
        
        # Plot a few examples
        num_examples = 3
        plt.figure(figsize=(12, 4 * num_examples))
        for i in range(num_examples):
            plt.subplot(num_examples, 3, i * 3 + 1)
            plt.imshow(noisy_images[i], cmap="gray")
            plt.title("Noisy Image")
            plt.axis("off")
            
            plt.subplot(num_examples, 3, i * 3 + 2)
            plt.imshow(predicted_images[i], cmap="gray")
            plt.title("Denoised Image")
            plt.axis("off")
            
            plt.subplot(num_examples, 3, i * 3 + 3)
            plt.imshow(clean_images[i], cmap="gray")
            plt.title("Clean Image")
            plt.axis("off")
        
        plt.tight_layout()
        plot_path = Path("docs/plots/denoising_results.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Denoising results saved to {plot_path}")
        
        self.save_score()

    def save_score(self):
        """Save evaluation metrics to a JSON file."""
        scores = {
            "test_mse": float(self.score),  # Convert to float for JSON serialization
            "average_psnr": float(self.avg_psnr),
            "average_ssim": float(self.avg_ssim)
        }
        save_json(path=Path("scores.json"), data=scores)
        logger.info("Evaluation scores saved to scores.json")

    def log_into_mlflow(self):
        """Log metrics, parameters, and artifacts to MLflow."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "test_mse": self.score,
                "average_psnr": self.avg_psnr,
                "average_ssim": self.avg_ssim
            })
            mlflow.log_artifact("docs/plots/denoising_results.png")
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="AutoencoderModel")
            else:
                mlflow.keras.log_model(self.model, "model")
        logger.info("Logged evaluation results to MLflow")