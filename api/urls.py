
from django.urls import include, path
from rest_framework import routers
from django.conf.urls.static import static
from . import views
from SemanticSegmentation import settings

app_name = 'api'

urlpatterns = [
    path(r'test', views.test_api, name='test_api_communication'),
    path(r'run/', views.RetrieveImages.as_view()),
    path(r'original/', views.original_results, name='return_original_segmentation'),
    path(r'shape/', views.return_shape, name='return_original_output_shape')

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)   