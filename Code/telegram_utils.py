import requests

# Function to read bot token and chat ID from a file
def read_telegram_config(file_path='telegram_config.txt'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    config = {}
    for line in lines:
        key, value = line.strip().split('=')
        config[key] = value
    return config['BOT_TOKEN'], config['CHAT_ID']

# Function to send a Telegram notification
def send_telegram_notification(message, config_file='telegram_config.txt'):
    bot_token, chat_id = read_telegram_config(config_file)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.status_code

# Function to format the data dictionary into a message string
def format_message(data):
    message = "\n".join([f"{key}: {value}" for key, value in data.items()])
    return message
