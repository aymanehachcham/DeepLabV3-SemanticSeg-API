from django.shortcuts import render

# Create your views here.
from django.views.decorators.cache import never_cache
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ImageSegmentation
from .utils import *
import shutil
from .serializers import OutputImageSerializer


@api_view(['GET'])
@never_cache
def test_api(request):
    return Response({'response':"You are successfully connected to the Semantic Segmentation API"})

@api_view(['POST'])
@never_cache
def run_inference(request):
    file_ = request.FILES['image']  
    image = ImageSegmentation.objects.create(input_image=file_, name='image_%02d' % uuid.uuid1())
    RunDeepLabInference(image).save_output()
    serializer = OutputImageSerializer(image)
    return Response(serializer.data)

@api_view(['GET'])
@never_cache
def clean_folders(request):
    folder_input = 'Media/Input_image/'
    for filename in os.listdir(folder_input):
        file_path = os.path.join(folder_input, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder_output = 'Media/Output_image/'
    for filename in os.listdir(folder_output):
        file_path = os.path.join(folder_output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return Response({'response': "Media folders were cleaned up!!"})
