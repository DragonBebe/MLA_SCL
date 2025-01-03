import os
import torch
import logging  # Import logging module
from Contrastive_Learning import config_con
from Contrastive_Learning import train, set_loader, set_model, create_scheduler, LARS, save_best_model

def ensure_dir_exists(path):
    """
    Ensure that the directory exists. If it does not exist, create it.
    
    Args:
        path (str): Directory path to ensure exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)  # Create directory
        print(f"Created directory: {path}")

def setup_logging(log_dir):
    """
    Configure logging for the training process. Automatically creates the log directory if it does not exist.
    
    Args:
        log_dir (str): Path to the log directory.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create log directory
        print(f"Created log directory: {log_dir}")

    log_file = os.path.join(log_dir, "training.log")  # Define log file path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define log format
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()        # Log to console
        ]
    )
    logging.info("Logging initialized. Logs will be saved to: %s", log_file)

def main():
    """
    Main function for training the contrastive learning model. This function sets up the environment,
    prepares the data, initializes the model, and executes the training loop.
    """
    # Get configurations from the configuration file
    opt = config_con.get_config()

    # Set up logging
    setup_logging(opt['log_dir'])

    # Ensure the save directory exists
    ensure_dir_exists(opt['model_save_dir'])

    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else "cpu")
    opt['device'] = device

    # Create the data loader and model
    train_loader = set_loader(opt)  # Data loader
    model, criterion, device = set_model(opt)  # Model, loss function, and device setup

    # Initialize optimizer and learning rate scheduler
    optimizer = LARS(
        model.parameters(),
        lr=opt['learning_rate'],  # Initial learning rate
        momentum=0.9,
        weight_decay=1e-4,
        eta=0.001,
        epsilon=1e-8,
        min_lr=1e-6  # Minimum learning rate
    )
    scheduler = create_scheduler(optimizer, warmup_epochs=5, total_epochs=opt['epochs'])  # Learning rate scheduler

    # Training loop variables
    best_loss = float('inf')  # Track the best loss
    last_save_path = None  # Last saved model path
    save_root = "./saved_models/pretraining"  # Root directory for saving pretrained models

    # Start training loop
    for epoch in range(opt['epochs']):
        logging.info(f"Epoch [{epoch + 1}/{opt['epochs']}] started.")  # Log epoch start
        print(f"Epoch [{epoch + 1}/{opt['epochs']}]")
        
        # Perform training for the epoch
        epoch_loss = train(train_loader, model, criterion, optimizer, opt, device, epoch)

        # Update the scheduler
        scheduler.step()

        # Log training loss and learning rate
        logging.info(f"Epoch [{epoch + 1}/{opt['epochs']}]: Train Loss: {epoch_loss:.4f}, "
                     f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save the best-performing model
        best_loss, last_save_path = save_best_model(model, opt, epoch, epoch_loss, save_root, best_loss, last_save_path)

    # Log and print training completion
    logging.info(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Training complete. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
