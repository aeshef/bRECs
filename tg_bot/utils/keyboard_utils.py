from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup

def get_main_keyboard():
    keyboard = [
        [InlineKeyboardButton("📊 Мой портфель", callback_data='main_portfolio')],
        # [InlineKeyboardButton("🔄 Рекомендации", callback_data='main_rebalance')], # Убрали ребаланс как действие
        [InlineKeyboardButton("⚙️ Настройки", callback_data='main_settings')],
        [InlineKeyboardButton("❓ Помощь", callback_data='main_help')]
    ]
    return InlineKeyboardMarkup(keyboard)