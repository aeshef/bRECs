import logging
import base64
import io
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from telegram import Bot, InputFile, TelegramError
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def to_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Не удалось преобразовать {value} типа {type(value)} в float")
        return 0.0

def format_portfolio_message(weights: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Форматирует данные о портфеле в читаемый текстовый отчет, 
    отображая только ключевые показатели.
    """
    python_metrics = {}
    for key, value in metrics.items():
        python_metrics[key] = to_float(value)
    
    logger.info(f"Метрики: {python_metrics}")
    
    output = "```\n"
    output += "╔═══════════════════════════════════════════════════╗\n"
    output += "║                ИНВЕСТИЦИОННЫЙ ПОРТФЕЛЬ             ║\n"
    output += "╚═══════════════════════════════════════════════════╝\n\n"
    
    output += "┌─────────────── КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ───────────────┐\n"
    
    key_metrics = {
        'expected_return': ['expected_return', 'annual_return', 'ожидаемая_доходность'],
        'expected_volatility': ['expected_volatility', 'volatility', 'annual_volatility', 'ожидаемая_волатильность'],
        'sharpe_ratio': ['sharpe_ratio', 'коэффициент_шарпа'],
        'max_drawdown': ['max_drawdown', 'максимальная_просадка'],
        'win_rate': ['win_rate', 'процент_выигрышных_периодов']
    }
    
    metric_labels = {
        'expected_return': 'Ожидаемая доходность',
        'expected_volatility': 'Ожидаемая волатильность',
        'sharpe_ratio': 'Коэффициент Шарпа',
        'max_drawdown': 'Максимальная просадка',
        'win_rate': 'Процент выигрышных периодов'
    }
    
    if not python_metrics:
        output += "│ Данные о метриках отсутствуют                  │\n"
    else:
        for key, possible_names in key_metrics.items():
            value = None
            for name in possible_names:
                if name in python_metrics:
                    value = python_metrics[name]
                    break
            
            if value is not None:
                label = metric_labels[key]
                if key in ['expected_return', 'expected_volatility', 'max_drawdown', 'win_rate']:
                    value_pct = value * 100
                    output += f"│ {label:<30} │ {value_pct:>6.2f}% │\n"
                else:
                    output += f"│ {label:<30} │ {value:>6.2f}  │\n"
    
    output += "└───────────────────────────────────────────────────┘\n"
    output += "```"
    
    return output

def create_portfolio_pie_chart(weights: Dict[str, Any]):
    """
    Создает круговую диаграмму портфеля с улучшенным форматированием.
    """
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        python_weights = {}
        for ticker, weight in weights.items():
            python_weights[ticker] = to_float(weight)
        
        filtered_weights = {k: v for k, v in python_weights.items() if v > 0.01}
        
        if not filtered_weights:
            logger.warning("No significant weights to display in portfolio pie chart")
            return None
        
        if len(filtered_weights) > 10:
            sorted_weights = sorted(filtered_weights.items(), key=lambda x: x[1], reverse=True)
            main_weights = dict(sorted_weights[:9])
            other_sum = sum(v for k, v in sorted_weights[9:])
            if other_sum > 0:
                main_weights['Другие'] = other_sum
            filtered_weights = main_weights

        fig, ax = plt.subplots(figsize=(12, 9))
        
        labels = []
        display_labels = []
        for ticker, weight in filtered_weights.items():
            if ticker == 'RISK_FREE':
                display_labels.append('Б/р активы')
                labels.append(f'Безрисковые активы ({weight*100:.1f}%)')
            else:
                # Для остальных тикеров просто берем название
                display_labels.append(ticker)
                labels.append(f'{ticker} ({weight*100:.1f}%)')

        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors
        
        wedges, texts = ax.pie(
            filtered_weights.values(),
            labels=None,
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.8},
            shadow=True,
            radius=0.7,
            colors=colors[:len(filtered_weights)]
        )

        ax.legend(
            wedges, 
            labels,
            title="Активы портфеля",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,    # Тень
            title_fontsize=12
        )
        
        for i, (wedge, label) in enumerate(zip(wedges, display_labels)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))
            
            horizontalalignment = 'center'
            
            if filtered_weights[list(filtered_weights.keys())[i]] < 0.05:
                continue
                
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                horizontalalignment=horizontalalignment,
                verticalalignment='center',
                fontsize=9,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        ax.set_title('Распределение активов в портфеле', fontsize=16, pad=20)
        fig.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        
        total = sum(filtered_weights.values())
        ax.annotate(
            f'Всего активов: {len(filtered_weights)} | Общий вес: {total*100:.2f}%',
            xy=(0.5, -0.1),
            xycoords='axes fraction',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="gray", alpha=0.8)
        )
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        logger.error(f"Error creating portfolio pie chart: {e}")
        return None


def send_result_notification(
    bot: Bot,
    chat_id: int,
    success: bool,
    weights: Dict[str, Any] = None,
    metrics: Dict[str, Any] = None,
    is_initial: bool = False,
    significant_changes: bool = False,
    report: Dict[str, Any] = None
):
    """
    Отправляет пользователю уведомление о результатах формирования портфеля.
    """
    try:
        if not success:
            bot.send_message(
                chat_id=chat_id,
                text="❌ *Не удалось сформировать инвестиционный портфель*.\n\n"
                     "Возникла техническая ошибка при расчете оптимального портфеля. "
                     "Наши специалисты уже работают над устранением проблемы. "
                     "Пожалуйста, попробуйте повторить запрос позже.",
                parse_mode="Markdown"
            )
            return

        if not weights or not metrics:
            if not report or not report.get('weights') or not report.get('metrics'):
                bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ *Внимание: данные портфеля неполные*\n\n"
                         "Получена частичная информация о портфеле. "
                         "Пожалуйста, попробуйте сформировать портфель снова.",
                    parse_mode="Markdown"
                )
                return
            weights = report.get('weights', {})
            metrics = report.get('metrics', {})

        header = "🚀 Ваш стартовый инвестиционный портфель готов!" if is_initial else "🔄 Ваш инвестиционный портфель обновлен!"
        
        if not is_initial and not significant_changes:
            bot.send_message(
                chat_id=chat_id,
                text=f"ℹ️ *Анализ вашего портфеля завершен*\n\n"
                     f"Существенных изменений в оптимальном составе портфеля не обнаружено. "
                     f"Текущий портфель соответствует рыночной ситуации и вашему профилю риска.",
                parse_mode="Markdown"
            )
            return

        main_text = f"{header}\n\n"
        
        if report and report.get('text_report'):
            portfolio_text = format_portfolio_message(weights, metrics)
            
            bot.send_message(
                chat_id=chat_id,
                text=main_text + portfolio_text,
                parse_mode="Markdown"
            )
        else:
            portfolio_text = format_portfolio_message(weights, metrics)
            bot.send_message(
                chat_id=chat_id,
                text=main_text + portfolio_text,
                parse_mode="Markdown"
            )

        pie_chart = create_portfolio_pie_chart(weights)
        if pie_chart:
            bot.send_photo(
                chat_id=chat_id,
                photo=InputFile(pie_chart),
                caption='📊 Распределение активов в портфеле'
            )

        if report and report.get('recommendations'):
            recommendations_text = "*💡 Рекомендации:*\n\n"
            for rec in report['recommendations']:
                recommendations_text += f"• {rec}\n"
            
            bot.send_message(
                chat_id=chat_id,
                text=recommendations_text,
                parse_mode="Markdown"
            )

        if report and report.get('images'):
            send_portfolio_images(bot, chat_id, report['images'])

    except TelegramError as te:
        logger.error(f"TelegramError while sending notification: {te}")
    except Exception as e:
        logger.exception(f"Error sending result notification: {e}")

def send_portfolio_images(bot: Bot, chat_id: int, images: Dict[str, str]):
    """
    Отправляет визуализации портфеля пользователю.
    """
    priority_order = [
        'efficient_frontier', 'cumulative_returns', 
        'monthly_calendar', 'drawdown_chart'
    ]
    
    image_titles = {
        'efficient_frontier': '📈 Эффективная граница портфеля',
        'cumulative_returns': '📈 Кумулятивная доходность',
        'monthly_calendar': '📅 Календарь месячной доходности',
        'drawdown_chart': '📉 График просадок портфеля'
    }
    
    sent_images = 0
    for image_type in priority_order:
        if image_type in images and sent_images < 2:
            try:
                caption = image_titles.get(image_type, 'График портфеля')
                img_data = base64.b64decode(images[image_type])
                img_bytesio = io.BytesIO(img_data)
                img_bytesio.name = f"{image_type}.png"
                
                bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(img_bytesio),
                    caption=caption
                )
                sent_images += 1
            except Exception as e:
                logger.error(f"Error sending image {image_type}: {e}")
