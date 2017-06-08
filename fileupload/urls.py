# encoding: utf-8
from django.conf.urls import url
from fileupload.views import (
        
        MusicDeleteView,MultiSvm,
        info,MusicCreateView,
        )
from . import views

urlpatterns = [
    
    url(r'^help/$', info.as_view(), name='info'),  
    url(r'^musicUpload/$', MusicCreateView.as_view(), name='upload-music'),  
    url(r'^$', MultiSvm.as_view(), name='upload'),   
    url(r'^delete/(?P<pk>\d+)$', MusicDeleteView.as_view(), name='upload-delete'), 
    url(r'^svm/$', views.music_genre, name='music_genre'),
    url(r'^multisvm/$', views.multi_music_genre, name='multi_music_genre'),
    

]
