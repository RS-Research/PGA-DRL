# Using Telegram Bot for Notifications

This guide explains how to configure and use the Telegram bot to send messages to a Telegram channel whenever results are produced in this repository.

## Prerequisites

1. **Telegram Bot**: Create a Telegram bot using [BotFather](https://t.me/BotFather) and obtain the `BOT_TOKEN`.
2. **Telegram Channel**: Create a channel and add the bot as an admin to get the `CHAT_ID`.

## Configuration

1. Create a file named `telegram_config.txt` in the root directory of the repository and add the following details:

   ```plaintext
   BOT_TOKEN=Your-Telegram-Bot-Token
   CHAT_ID=The-Telegram-Channel-Chat-ID
   ```

   Example:
   ```plaintext
   BOT_TOKEN=...
   CHAT_ID=...
   ```

2. Ensure the bot has admin privileges in the specified Telegram channel.

## Sending Notifications

### Python Script
The script `telegram_utils.py` contains functions to send messages using the Telegram Bot API. Here's how it works:

#### Import the Utility
Ensure you have the `telegram_utils.py` file in your project and import it:

```python
from telegram_utils import send_telegram_notification, format_message
```

#### Sending a Notification
You can send a message by calling the `send_telegram_notification` function:

```python
message = "Your custom notification message"
send_telegram_notification(message)
```

Alternatively, if you have data to format:

```python
data = {
    "Result": "Success",
    "Model": "PGA-DRL",
    "Metric": "Precision@10",
    "Value": "0.2155"
}
message = format_message(data)
send_telegram_notification(message)
```

#### Return Value
The `send_telegram_notification` function returns the HTTP status code of the Telegram API response. A `200` status code indicates a successful message delivery.

## Example Workflow

1. Ensure the `telegram_config.txt` file is correctly configured.
2. Use the script to send messages whenever results are produced, e.g., after model training or evaluation.
3. Monitor your Telegram channel for notifications.

## Notes
- Make sure the `requests` library is installed:
  ```bash
  pip install requests
  ```
- Protect your `telegram_config.txt` file to ensure your bot token and channel ID remain secure.

## License
This bot integration follows the repository's [MIT License](LICENSE).
