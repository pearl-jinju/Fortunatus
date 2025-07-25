# stock_analyzer/admin.py
from django.contrib import admin
from .models import Stock, StockDailyPrice

# Stock 모델 관리자 설정
class StockAdmin(admin.ModelAdmin):
    # list_display 항목을 모델의 실제 필드 이름인 'name'과 'market'으로 변경
    list_display = ('symbol', 'name', 'market') # <-- 이 부분을 변경합니다.
    search_fields = ('symbol', 'name') # <-- 이 부분도 변경합니다.

# StockDailyPrice 모델 관리자 설정
class StockDailyPriceAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'close_price', 'volume')
    list_filter = ('stock__symbol', 'date')
    search_fields = ('stock__symbol',)
    date_hierarchy = 'date'

admin.site.register(Stock, StockAdmin)
admin.site.register(StockDailyPrice, StockDailyPriceAdmin)