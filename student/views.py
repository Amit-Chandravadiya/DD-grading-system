from operator import imod
from django.shortcuts import render
import numpy as np
import os
import joblib

# Create your views here.

def show(request):
    return render(request,"Marks.html")

def pred(request):
    if request.method=="POST":
        study_time=int(request.POST['mark'])
        health=int(request.POST['status'])
        absence=int(request.POST['absence'])
        failures=int(request.POST['failures'])
        G1=int(request.POST['G1'])
        G2=int(request.POST['G2'])
        
        arr=np.array([[study_time,health,absence,failures,G1,G2]])
        print(arr)
        
        cwd=os.getcwd()
        loc=os.path.join(cwd,'student/Dhairya_knn.pkl')
        print(loc)
        
        model=joblib.load(loc)
        res=model.predict(arr)
    context={
        "predict":res[0]
    }
    return render(request,"Marks.html",context)
