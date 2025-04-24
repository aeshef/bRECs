import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def get_kbd(start_date, end_date):
    # Установим URL для парсинга
    url = "https://cbr.ru/hd_base/zcyc_params/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    # Создаем сессию, которая будет использовать cookies
    session = requests.Session()
    session.headers.update(headers)

    # Пробуем сделать запрос
    response = session.get(url)

    # Если запрос прошел успешно
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Поиск таблицы, в которой содержатся нужные данные
        table = soup.find('table', {'class': 'data spaced'})  # Убедитесь, что это правильный класс

        # Если таблица найдена, извлекаем данные
        if table:
            # Извлекаем заголовки столбцов (первая строка таблицы)
            headers = []
            header_row = table.find_all('tr')[1]  # Заголовки вторая строка (вторая строка - по 12 срокам)
            header_columns = header_row.find_all('th')[1:]  # Пропускаем первый столбец (Дата)
            for col in header_columns:
                headers.append(col.text.strip())

            # Преобразуем таблицу в pandas DataFrame
            rows = table.find_all('tr')[2:]  # Пропускаем заголовки
            data = []

            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 1:
                    date_str = cols[0].text.strip()
                    # Преобразуем дату в формат datetime
                    date_obj = datetime.strptime(date_str, '%d.%m.%Y')

                    # Если дата входит в указанный диапазон, добавляем данные в список
                    if start_date <= date_obj <= end_date:
                        row_data = {'date': date_obj}
                        
                        # Проверяем, что количество значений в строке соответствует количеству заголовков
                        for i, col in enumerate(cols[1:]):
                            if i < len(headers):  # Проверка, чтобы избежать IndexError
                                row_data[headers[i]] = col.text.strip()

                        data.append(row_data)

            # Преобразуем список в DataFrame
            df = pd.DataFrame(data)

            # Конвертируем в CSV
            df.to_csv('/Users/aeshef/Documents/GitHub/kursach/data/processed_data/kbd.csv', index=False)
            print(f"Данные успешно сохранены в 'kbd_data.csv'")

        else:
            print("Таблица не найдена на странице.")
    else:
        print(f"Ошибка запроса: {response.status_code}")
