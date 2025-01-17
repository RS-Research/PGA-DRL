from telegram_utils import send_telegram_notification
from recbole.quick_start import run_recbole
import yaml
import os

# Define the initial message for the Telegram notification
message = 'Paper Evaluation - Server Run ==> PGA-DRL: Progressive Graph Attention-Based Deep Reinforcement Learning for Recommender Systems'
send_telegram_notification(message)

# Path to the configuration file
config_file_path = 'config/pga_drl_config.yaml'

# Load the YAML configuration file dynamically
with open(config_file_path, 'r') as file:
    config_settings = yaml.safe_load(file)

# Extract relevant settings for the message
dataset = config_settings.get('dataset', 'ml-100k')
lr = config_settings.get('learning_rate', 'Not specified')
epochs = config_settings.get('epochs', 'Not specified')
embedding_size = config_settings.get('embedding_size', 'Not specified')
n_hops = config_settings.get('num_gcn_layers', 'Not specified')
batch_size = config_settings.get('train_batch_size', 'Not specified')
optimizer = config_settings.get('optimizer', 'Not specified')
eval_metrics = config_settings.get('metrics', 'Not specified')

# Create the settings message dynamically
settings_message = (
    f"Running Model: PGA-DRL\n"
    f"Dataset: {dataset}\n"
    f"Learning Rate: {lr}\n"
    f"Epochs: {epochs}\n"
    f"Embedding Size: {embedding_size}\n"
    f"Number of GCN and GAT layers (n_hops): {n_hops}\n"
    f"Batch Size: {batch_size}\n"
    f"Optimizer: {optimizer}\n"
    f"Evaluation Metrics: {', '.join(eval_metrics) if eval_metrics else 'Not specified'}"
)

# Send dynamic settings via Telegram
send_telegram_notification(settings_message)

# Run the model
result_metrics = run_recbole(model='PGA_DRL', dataset='ml-100k', config_file_list=[config_file_path])

# Send result metrics via Telegram
send_telegram_notification(f"Training completed! Result metrics:\n{str(result_metrics)}")
