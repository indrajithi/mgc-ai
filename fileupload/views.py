# encoding: utf-8
import json

from django.http import HttpResponse
from django.views.generic import CreateView, DeleteView, ListView, DetailView
from .models import Picture, Music
from .response import JSONResponse, response_mimetype
from .serialize import serialize
import json
import random
from mysvm import feature
from mysvm import svm



class MusicCreateView(CreateView):
    model = Music
    fields = "__all__"

    def form_valid(self, form):
        self.object = form.save()
        files = [serialize(self.object)]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response

    def form_invalid(self, form):
        data = json.dumps(form.errors)
        return HttpResponse(content=data, status=400, content_type='application/json')



class info(MusicCreateView):
    template_name_suffix = '_svm_info'


class MultiSvm(MusicCreateView):
    template_name_suffix = '_svm_multi'



class MusicDeleteView(DeleteView):
    model = Music

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()
        self.object.delete()
        response = JSONResponse(True, mimetype=response_mimetype(request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response

def music_genre(request):
    model = Music

    if request.method == 'POST':
        #context = feature.getlabels()
        context = ['Classical','Hipop','Jazz','Metal','Pop','Rock']
        #return (request,"love")
        try:
            JSONdata = json.loads(str(request.body, encoding="utf-8"))
        except:
            JSONdata = 'ERROR'
        
        
        #get index of the genre
        genre = svm.getGenre(JSONdata['file'])

        #delete file after finding genre
        id = JSONdata['delete']
        instance = model.objects.get(id=id)
        instance.delete()

        return HttpResponse(context[int(genre[0]) - 1 ])
    if request.method == 'GET':
        return HttpResponse('nothing here')




def multi_music_genre(request):
    model = Music
    if request.method == 'POST':
        
        try:
            JSONdata = json.loads(str(request.body, encoding="utf-8"))
        except:
            JSONdata = 'ERROR'
        print(JSONdata['file'])

        #get index of the genre
        dd, genre = svm.getMultiGenre(JSONdata['file'])
        print(dd)
        dt = json.dumps(dd)

        #delete file after finding genre
        id = JSONdata['delete']
        instance = model.objects.get(id=id)
        instance.delete()

        return HttpResponse(', '.join(genre))
        #return HttpResponse(dt)

    if request.method == 'GET':
        return HttpResponse('nothing here')


class PictureListView(ListView):
    model = Music

    def render_to_response(self, context, **response_kwargs):
        files = [ serialize(p) for p in self.get_queryset() ]
        data = {'files': files}
        response = JSONResponse(data, mimetype=response_mimetype(self.request))
        response['Content-Disposition'] = 'inline; filename=files.json'
        return response
