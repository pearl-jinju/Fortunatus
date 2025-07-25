from django.contrib import admin
from django.urls import path, include
from stock_analyzer.views import stock_analysis_page # 새로 추가할 뷰를 임포트

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/analyze-strategy/', include('stock_analyzer.urls')), # 기존 API URL
    path('stock-analysis/', stock_analysis_page, name='stock_analysis_page'), # 새로 추가할 웹 페이지 URL
]