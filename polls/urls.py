#-*-coding:utf-8
from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = {
    path('', views.index, name="index"),
    path('1/', views.index1, name="po"),
    path('2', views.test, name="po1"),
}