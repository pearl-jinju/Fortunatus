import yfinance as yf
from datetime import datetime, date, timedelta

from django.core.management.base import BaseCommand, CommandError
from stock_analyzer.models import Stock, StockDailyPrice

class Command(BaseCommand):
    help = 'Fetches historical daily stock data using yfinance and stores it.'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='The stock symbol (e.g., "005930" for KOSPI, "AAPL" for NASDAQ).', required=True)
        parser.add_argument('--start_date', type=str, help='Start date for fetching data (YYYY-MM-DD). Default: 2000-01-01.')
        parser.add_argument('--end_date', type=str, help='End date for fetching data (YYYY-MM-DD). Default: today.')
        parser.add_argument('--full_sync', action='store_true', help='Fetch data from a very early date (e.g., 2000-01-01) up to today. Overrides --start_date/--end_date.')

    def handle(self, *args, **options):
        original_symbol = options['symbol'] # 사용자가 입력한 원래 심볼
        full_sync = options['full_sync']
        start_date_str = options['start_date']
        end_date_str = options['end_date']

        # --- 심볼 형식 처리 (한국 주식 vs. 해외 주식) ---
        # 사용자가 입력한 원래 심볼을 DB에 저장하고, yfinance용 심볼은 따로 준비합니다.
        # 한국 주식 (6자리 숫자)인 경우 .KS 또는 .KQ를 붙입니다.
        if len(original_symbol) == 6 and original_symbol.isdigit():
            # 사용자가 이미 .KS/.KQ를 붙여줬는지 확인하는 것도 좋지만, 일단 기본적으로 KOSPI로 가정합니다.
            # 만약 KOSDAQ 주식이라면 '035720.KQ'처럼 직접 입력해야 합니다.
            symbol_for_yf = f"{original_symbol}.KS"
            self.stdout.write(self.style.WARNING(f"Assuming '{original_symbol}' is a KOSPI stock. Using yfinance symbol: '{symbol_for_yf}'"))
        else:
            # 그 외의 경우 (영문 심볼, 이미 .KS/.KQ가 붙은 경우 등)는 그대로 사용
            symbol_for_yf = original_symbol
            self.stdout.write(f"Using yfinance symbol: '{symbol_for_yf}'")

        # 종목 존재 여부 확인 및 생성/업데이트
        # DB에는 사용자가 입력한 원래 심볼을 저장합니다. (예: 'AAPL' 또는 '005930')
        stock, created = Stock.objects.get_or_create(
            symbol=original_symbol,
            defaults={'company_name': f'Unknown ({original_symbol})'}
        )
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created new stock entry for {original_symbol}.'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Found existing stock entry for {original_symbol}.'))

        # --- 데이터 수집 시작/종료 날짜 설정 ---
        if full_sync:
            fetch_start_date = date(1900, 1, 1) # 아주 오래된 과거부터
            fetch_end_date = date.today()
            self.stdout.write(f"Fetching FULL historical data for {symbol_for_yf} from {fetch_start_date} to {fetch_end_date}...")
        else:
            # 마지막 저장된 날짜 이후부터 가져오기
            last_saved_date = None
            try:
                last_price = StockDailyPrice.objects.filter(stock=stock).order_by('-date').first()
                if last_price:
                    last_saved_date = last_price.date
                    # 마지막 저장된 날짜 다음 날부터 가져오기 시작
                    fetch_start_date = last_saved_date + timedelta(days=1)
                    self.stdout.write(f"Last saved date for {original_symbol}: {last_saved_date}. Fetching data from {fetch_start_date}...")
                else:
                    fetch_start_date = date(2000, 1, 1) # 저장된 데이터 없으면 2000년부터
                    self.stdout.write(f"No previous data found for {original_symbol}. Fetching from {fetch_start_date}...")

            except StockDailyPrice.DoesNotExist:
                fetch_start_date = date(2000, 1, 1)
                self.stdout.write(f"No previous data found for {original_symbol}. Fetching from {fetch_start_date}...")

            fetch_end_date = date.today() # 오늘까지

            # `--start_date` 및 `--end_date` 옵션이 주어진 경우 오버라이드
            if start_date_str:
                fetch_start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            if end_date_str:
                fetch_end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

            self.stdout.write(f"Fetching data for {symbol_for_yf} from {fetch_start_date} to {fetch_end_date}...")


        # yfinance를 사용하여 데이터 다운로드
        try:
            ticker = yf.Ticker(symbol_for_yf)
            # ticker.history는 주말을 포함하여 날짜를 요청하지만, 실제 거래일만 반환.
            # start/end 날짜 지정 시, 끝 날짜는 포함되지 않으므로 하루 더 추가하는 것이 일반적
            # 또는 end=datetime.now() 등으로 오늘까지 지정. 여기서는 fetch_end_date가 끝날이므로
            # 다음날까지 포함하도록 +python manage.py fetch_stock_data --symbol AAPL --full_sync timedelta(days=1)를 해줍니다.
            data_df = ticker.history(start=fetch_start_date.strftime('%Y-%m-%d'),
                                     end=(fetch_end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                                     interval="1d")

            if data_df.empty:
                self.stdout.write(self.style.WARNING(f"No data fetched for {symbol_for_yf} between {fetch_start_date} and {fetch_end_date}."))
                return

            prices_saved_count = 0
            for index, row in data_df.iterrows():
                # 인덱스가 datetime 객체이므로 date()로 변환
                current_date = index.date()

                # yfinance 데이터 컬럼명: Open, High, Low, Close, Volume, Dividends, Stock Splits
                try:
                    obj, created = StockDailyPrice.objects.update_or_create(
                        stock=stock,
                        date=current_date,
                        defaults={
                            'open_price': round(row['Open'], 2),
                            'high_price': round(row['High'], 2),
                            'low_price': round(row['Low'], 2),
                            'close_price': round(row['Close'], 2),
                            'volume': int(row['Volume'])
                        }
                    )
                    if created:
                        prices_saved_count += 1

                except (ValueError, KeyError) as e:
                    self.stdout.write(self.style.ERROR(f"Error processing data for {symbol_for_yf} on {current_date}: {e}"))
                    continue

            self.stdout.write(self.style.SUCCESS(f'Successfully saved/updated {prices_saved_count} new daily prices for {original_symbol}.'))

        except Exception as e:
            raise CommandError(f"Error fetching data with yfinance for {symbol_for_yf}: {e}")
