from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyze_stock_strategy, name='analyze_stock_strategy'),
]