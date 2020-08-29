from django.shortcuts import render

# Create your views here.
from django.views.decorators.cache import never_cache
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ImageSegmentation
from .utils import *
import shutil
from rest_framework.views import APIView
from .serializers import OutputImageSerializer
from rest_framework import status


@api_view(['GET'])
@never_cache
def test_api(request):
    return Response({'response':"You are successfully connected to the Semantic Segmentation API"})

@api_view(['POST'])
@never_cache
def original_results(request):
    file_ = request.FILES['image']
    image = ImageSegmentation.objects.create(input_image=file_, name='image_%02d' % uuid.uuid1())
    RunDeepLabInference(image).save_original_output()
    serializer = OutputImageSerializer(image)
    return Response(serializer.data, status=status.HTTP_200_OK)

class RetrieveImages(APIView):

    def delete(self, request):
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

    def post(self, request):
        file_ = request.FILES['image']
        image = ImageSegmentation.objects.create(input_image=file_, name='image_%02d' % uuid.uuid1())
        RunDeepLabInference(image).save_output()

        serializer = OutputImageSerializer(image)
        return Response(serializer.data, status=status.HTTP_200_OK)



