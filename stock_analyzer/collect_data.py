# stock_analyzer/data_collector.py
import FinanceDataReader as fdr
import pandas as pd
from datetime import date, timedelta

from .models import Stock, StockDailyPrice

def collect_stock_data(symbol, start_date=None, end_date=None):
    if not start_date:
        start_date = (date.today() - timedelta(days=365*5)).strftime('%Y-%m-%d') # 기본 5년치
    if not end_date:
        end_date = date.today().strftime('%Y-%m-%d')

    try:
        # FinanceDataReader를 사용하여 종목 정보 가져오기 시도
        # PLTR 같은 해외 주식은 StockListing에 없을 가능성이 높으므로 일반적인 예외 처리
        stock_name = symbol # 기본값: 심볼을 이름으로 사용
        market_name = "OVERSEAS" # 기본값: 해외 시장으로 가정

        try:
            # FinanceDataReader StockListing은 주로 국내 주식 정보 제공
            # 해외 주식은 여기서 찾지 못할 가능성이 높음. (PLTR은 미국 주식)
            # 따라서 이 부분은 그대로 두되, 오류 발생 시 아래 except 블록에서 처리
            stock_info_df = fdr.StockListing('NASDAQ') # 예시: NASDAQ 리스트를 시도
            if symbol in stock_info_df['Symbol'].values:
                info = stock_info_df[stock_info_df['Symbol'] == symbol].iloc[0]
                stock_name = info['Name']
                market_name = info['Market']
            else:
                 # 'Symbol'이나 'NASDAQ'에서 못 찾으면 기본값 사용
                print(f"FinanceDataReader StockListing에서 '{symbol}' 종목 정보를 찾을 수 없습니다. 기본값을 사용합니다.")
        except Exception as e:
            print(f"FinanceDataReader StockListing 과정에서 오류 발생 ({e}). 종목 코드를 이름으로 사용합니다.")
            # 오류 발생 시: stock_name과 market_name은 위에서 설정한 기본값 유지

        # Stock 모델에 저장 또는 가져오기
        # 'code' 대신 'symbol' 필드 사용. name과 market도 함께 전달
        stock, created = Stock.objects.get_or_create(
            symbol=symbol,
            defaults={'name': stock_name, 'market': market_name} # <-- defaults 인자로 name, market 전달
        )
        if not created: # 이미 존재하는 경우에도 이름과 시장 업데이트 (선택 사항)
            if stock.name != stock_name or stock.market != market_name:
                stock.name = stock_name
                stock.market = market_name
                stock.save()
                print(f"'{symbol}' 종목 정보 업데이트: 이름='{stock_name}', 시장='{market_name}'")


        # 주가 데이터 가져오기
        df = fdr.DataReader(symbol, start=start_date, end=end_date)
        if df.empty:
            return {'error': f'"{symbol}" 종목의 데이터를 FinanceDataReader에서 찾을 수 없습니다.'}

        # 데이터베이스에 저장
        for _, row in df.iterrows():
            StockDailyPrice.objects.update_or_create(
                stock=stock,
                date=row.name.date(), # 인덱스가 날짜/시간 객체이므로 .date()를 사용하여 날짜만 추출
                defaults={
                    'open_price': row['Open'],
                    'high_price': row['High'],
                    'low_price': row['Low'],
                    'close_price': row['Close'],
                    'volume': row['Volume']
                }
            )
        return {'message': f'"{symbol}" 종목 데이터가 성공적으로 수집되었습니다.'}

    except Exception as e:
        return {'error': f'데이터 수집 중 오류 발생: {str(e)}'}