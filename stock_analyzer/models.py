# stock_analyzer/models.py
from django.db import models

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True) # <-- name 필드 추가
    market = models.CharField(max_length=50, blank=True, null=True) # <-- market 필드 추가

    def __str__(self):
        return self.symbol

class StockDailyPrice(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('stock', 'date')
        ordering = ['date']

    def __str__(self):
        return f"{self.stock.symbol} - {self.date}"