"""
Demo code

Example with single image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg
```
"""

import torch
import torchvision
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import PIL
from PIL import Image
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from models import hmr, SMPL
from utils.imutils import crop
import config
import constants


class PersonDetector:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        self.model = model.to(self.device)

    def crop_person_box(self, frame, person_index=0):
        image_tensor = torchvision.transforms.functional.to_tensor(frame).to(self.device)
        with torch.no_grad():
            output = self.model([image_tensor])[0]

        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        for i, label in enumerate(labels):
            if label == 1:
                x1, x2, y1, y2 = boxes[i]
                break

        person_img = frame.crop((x1, x2, y1, y2))

        return person_img

    def process_image(self, img_file, input_res=224):
        """Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
        img = Image.fromarray(img).convert('RGB')
        img = self.crop_person_box(img)
        img.save("output/crop.jpg", "JPEG")
        img = np.array(img)

        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200

        img = crop(img, center, scale, (input_res, input_res))
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        norm_img = normalize_img(img.clone())[None]
        return img, norm_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--img', type=str, required=True, help='Path to input image')

    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    detector = PersonDetector()

    # Preprocess input image and generate predictions
    _, norm_img = detector.process_image(args.img, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

    pred_mesh = pred_vertices[0].cpu().numpy()

    fig = pyplot.figure()
    ax = Axes3D(fig)

    ax.scatter(pred_mesh[:, 1], pred_mesh[:, 2], pred_mesh[:, 0])

    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
    pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
    # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
    pred_keypoints_3d = pred_keypoints_3d[:, constants.H36M_TO_J14, :]
    # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

    joints = pred_keypoints_3d[0].cpu().numpy()

    ax.scatter(joints[:, 1], joints[:, 2], joints[:, 0], color='r')
    pyplot.savefig('output/temp.png')
