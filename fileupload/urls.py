# encoding: utf-8
from django.conf.urls import url
from fileupload.views import (
        BasicVersionCreateView, BasicPlusVersionCreateView,
        jQueryVersionCreateView, AngularVersionCreateView,
        PictureCreateView, PictureDeleteView, PictureListView,MultiSvm,
        info,
        )
from . import views

urlpatterns = [
    url(r'^basic/$', BasicVersionCreateView.as_view(), name='upload-basic'),
    url(r'^help/$', info.as_view(), name='info'),
    url(r'^basic/plus/$', BasicPlusVersionCreateView.as_view(), name='upload-basic-plus'),
    #url(r'^new/$', PictureCreateView.as_view(), name='upload-new'),
    url(r'^angular/$', AngularVersionCreateView.as_view(), name='upload-angular'),
    url(r'^multi/$', MultiSvm.as_view(), name='upload-multi'),
    url(r'^$', AngularVersionCreateView.as_view(), name='upload-angular'),
    url(r'^jquery-ui/$', jQueryVersionCreateView.as_view(), name='upload-jquery'),
    url(r'^delete/(?P<pk>\d+)$', PictureDeleteView.as_view(), name='upload-delete'),
    #url(r'^svm/(?P<pk>\d+)$', views.music_genre, name='music_genre'),
    url(r'^svm/$', views.music_genre, name='music_genre'),
    url(r'^multisvm/$', views.multi_music_genre, name='multi_music_genre'),
    url(r'^view/$', PictureListView.as_view(), name='upload-view'),

]
