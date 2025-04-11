import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

class FinamFinancialParser:
    def __init__(self, ticker, base_path='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
        self.ticker = ticker
        self.base_url = f'https://www.finam.ru/quote/moex/{ticker.lower()}/financial/'
        self.save_path = os.path.join(base_path, ticker.upper())

    def fetch_html(self):
        response = requests.get(self.base_url)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"Failed to retrieve data for ticker {self.ticker}. HTTP Status Code: {response.status_code}")
            return None
    
    def get_tables(self, soup):
        headers = soup.find_all(['h2', 'h3'])
        tables = soup.find_all('table')
        
        dataframes = []
        
        for header, table in zip(headers, tables):
            table_title = header.text.strip()
            rows = table.find_all('tr')
            data = []

            for row in rows:
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                if cols:
                    data.append(cols)
            
            if data:
                headers = ['Показатель'] + [f'Период {i}' for i in range(1, len(data[0]))]
                df = pd.DataFrame(data, columns=headers)
                df['Категория'] = table_title
                dataframes.append(df)
        
        return dataframes

    def parse_financial_data(self):
        soup = self.fetch_html()
        if soup is not None:
            tables = self.get_tables(soup)
            if tables:
                complete_df = pd.concat(tables, ignore_index=True, join='outer')
                complete_df['Показатель'] = complete_df['Показатель'].apply(lambda x: x.split('\xa0')[0])
                return complete_df
        return None

    def save_to_csv(self, df):
        os.makedirs(self.save_path, exist_ok=True)
        csv_file = os.path.join(self.save_path, f'{self.ticker}_financial_data.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Data saved to {csv_file}")