import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.models.segmentation as models
from SemanticModels.Dataset.semantic_data import SegmentationSample

class SemanticSeg(nn.Module):
    def __init__(self, pretrained: bool, device):
        super(SemanticSeg, self).__init__()
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        if device == 'cpu':
            self.device = 'cpu'

        self.model = self.load_model(pretrained)

    def __getitem__(self, item):
        return self.model

    # Add the Backbone option in the parameters
    def load_model(self, pretrained=False):
        if pretrained:
            model = models.deeplabv3_resnet101(pretrained=True)
        else:
            model = models.deeplabv3_resnet101()

        model.to(self.device)
        model.eval()
        return model

    def run_inference(self, image: SegmentationSample):
        # Run the model in the respective device:
        with torch.no_grad():
            output = self.model(image.processed_image)['out']

        reshaped_output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        return self.decode_segmentation(reshaped_output, image.image_file)


    def decode_segmentation(self, input_image, source, number_channels=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        # Defining empty matrices for rgb tensors:
        r = np.zeros_like(input_image).astype(np.uint8)
        g = np.zeros_like(input_image).astype(np.uint8)
        b = np.zeros_like(input_image).astype(np.uint8)

        for l in range(0, number_channels):
            if l == 15:
                idx = input_image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        # return rgb

        foreground = cv2.imread(source)

        # and resize image to match shape of R-band in RGB output map
        foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

        background = 0 * np.ones_like(rgb).astype(np.uint8)

        foreground = foreground.astype(float)
        background = background.astype(float)

        # Create a binary mask using the threshold
        th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

        # Apply Gaussian blur to the alpha channel
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        alpha = alpha.astype(float) / 255

        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)

        outImage = cv2.add(foreground, background)

        return outImage / 255
