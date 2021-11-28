from django.shortcuts import render
from django.http import HttpResponse
from .forms import AudioForm
from .input_audio_preprocess import save_mfcc
from .CNN import classify
import os

def index(request):
    if request.method=='POST':
        form = AudioForm(request.POST,request.FILES or None) 
        file=request.FILES['record'].name
        if form.is_valid() and file.endswith('.wav'): 
            form.save() 

            save_mfcc(os.path.dirname(os.path.realpath(__file__))+"/documents/"+file,
                json_path=os.path.dirname(os.path.realpath(__file__))+"/audio.json", 
                num_segments=10)
            
            genre=classify()

            return render(request,"classifier/result.html",{'genre':genre})
        else:
            return HttpResponse('Invalid file')

    else:    
        form = AudioForm() 
    return render(request, "classifier/base.html", {'form' : form}) 