
import os
import uuid
from PIL import Image
import numpy as np
from SemanticModels.Dataset.semantic_data import SegmentationSample
from SemanticModels.DeepLabV3.deeplab_implemen import SemanticSeg

def get_input_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'Media/Input_image/{}{}'.format(uuid.uuid4(), ext)

def get_output_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'Media/Output_image/{}{}'.format(uuid.uuid4(), ext)

class RunDeepLabInference():
    def __init__(self, image_file):
        self.file_image = image_file
        self.output_folder = 'Media/Output_image/'
        self.base_path, self.filename = os.path.split(self.file_image.input_image.path)
        self.sample_image = SegmentationSample(root_dir=self.base_path, image_file=self.filename, device='cuda')
        self.model = SemanticSeg(pretrained=True, device='cuda')

    def save_output(self):
        res = self.model.run_inference(self.sample_image)
        image_to_array = Image.fromarray((res * 255).astype(np.uint8))
        image_to_array.save(self.output_folder + self.filename)
        self.file_image.output_image = self.output_folder + self.filename
        self.file_image.save()