
import matplotlib.pyplot as plt
import cv2
from SemanticModels.Dataset.semantic_data import SegmentationSample
from SemanticModels.DeepLabV3.deeplab_implemen import SemanticSeg
import uuid


input_image = 'image-test.jpeg'
image = SegmentationSample(root_dir='Media/Input_image', image_file=input_image, device='cpu')
model = SemanticSeg(pretrained=True, device='cpu')

output = model.run_inference(image)
plt.imshow(output)
plt.pause(5)


#res = model(image)

#print(res.size())