# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import torch
import time
from PIL import Image
import numpy
from matplotlib import pyplot
def combine_masks_tuple(input_model):
    # type: (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,Tensor]) -> Tensor
    image_with_mask, bboxes, labels, masks, scores,palette=input_model
    threshold=0.5
    padding=1
    contour=True
    rectangle=False
    
    
    height = 800
    width = 800
    #image_with_mask = image.clone()
    for i in range(masks.size(0)):
        color = ((palette * labels[i]) % 255).to(torch.uint8)
        one_mask = my_paste_mask(masks[i, 0], bboxes[i], height, width, threshold, padding, contour, rectangle)
        image_with_mask = torch.where(one_mask.unsqueeze(-1), color.unsqueeze(0).unsqueeze(0), image_with_mask)
    return image_with_mask

def processImage(name,size, model):
    pil_image =Image.open(name).convert("RGB")
    pil_image = pil_image.resize((size, size), Image.BILINEAR)
    image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
    image = (image.float()).permute(2, 0, 1) - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
    ImageFinal = image.unsqueeze(0).to(model.device)
    return ImageFinal

def processImageCPU(name,size, model):
    image2 =Image.open(name).convert("RGB")
    image2 = image2.resize((size, size), Image.BILINEAR)
    image2 = torch.from_numpy(numpy.array(image2)[:, :, [2, 1, 0]])
    return image2

def my_paste_mask(mask, bbox, height, width, threshold=0.5, padding=1, contour=False, rectangle=False):
    # type: (Tensor, Tensor, int, int, float, int, bool, bool) -> Tensor
    padded_mask = torch.constant_pad_nd(mask, (padding, padding, padding, padding))
    #print("mask.size(-1)",mask.size(-1))
    scale = 1.0 + 2.0 * float(padding) / float(mask.size(-1))
    #print("scale",scale)
    center_x = (bbox[2] + bbox[0]) * 0.5
    center_y = (bbox[3] + bbox[1]) * 0.5
    w_2 = (bbox[2] - bbox[0]) * 0.5 * scale
    h_2 = (bbox[3] - bbox[1]) * 0.5 * scale  # should have two scales?
    bbox_scaled = torch.stack([center_x - w_2, center_y - h_2,
                               center_x + w_2, center_y + h_2], 0)

    TO_REMOVE = 1
    w = (bbox_scaled[2] - bbox_scaled[0] + TO_REMOVE).clamp(min=1).long()
    h = (bbox_scaled[3] - bbox_scaled[1] + TO_REMOVE).clamp(min=1).long()

    scaled_mask = torch.ops.maskrcnn_benchmark.upsample_bilinear(padded_mask.float(), h, w)
    
    x0 = bbox_scaled[0].long()
    y0 = bbox_scaled[1].long()
    x = x0.clamp(min=0)
    y = y0.clamp(min=0)
    #print("scaled_mask",scaled_mask.size())
    leftcrop = x - x0
    topcrop = y - y0
    w = torch.min(w - leftcrop, width - x)
    h = torch.min(h - topcrop, height - y)
    #print("h",h,"w",w)
    #mask = torch.zeros((height, width), dtype=torch.uint8)
    #mask[y:y + h, x:x + w] = (scaled_mask[topcrop:topcrop + h,  leftcrop:leftcrop + w] > threshold)
    mask = torch.constant_pad_nd((scaled_mask[topcrop:topcrop + h, leftcrop:leftcrop + w] > threshold),
                                 (int(x), int(width - x - w), int(y), int(height - y - h)))   # int for the script compiler

    if contour:
        mask = mask.float()
        # poor person's contour finding by comparing to smoothed
        mask = (mask - torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0),
                                                  torch.full((1, 1, 3, 3), 1.0 / 9.0), padding=1)[0, 0]).abs() > 0.001
    if rectangle:
        x = torch.arange(width, dtype=torch.long).unsqueeze(0)
        y = torch.arange(height, dtype=torch.long).unsqueeze(1)
        r = bbox.long()
        # work around script not liking bitwise ops
        rectangle_mask = ((((x == r[0]) + (x == r[2])) * (y >= r[1]) * (y <= r[3]))
                          + (((y == r[1]) + (y == r[3])) * (x >= r[0]) * (x <= r[2])))
        mask = (mask + rectangle_mask).clamp(max=1)
    #print(mask.size())
    return mask

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    start_time = time.time()
    image = processImage("test.jpg",800,coco_demo)
    image2 = processImageCPU("test.jpg",800,coco_demo)
    coco_demo.single_image_to_top_predictions(image)

    for p in coco_demo.model.parameters():
        p.requires_grad_(False)
    coco_demo.model = coco_demo.model.eval()
    with torch.jit.optimized_execution(False):
        traced_model = torch.jit.trace(coco_demo.single_image_to_top_predictions, image, check_trace=False)
    traced_model.save('traced.pt')

    print("done tracing")

    print("testing first image:")


    loaded = torch.jit.load("traced.pt")
    boxes, labels, masks, scores = loaded(image)
    palette=torch.tensor([3, 32767, 2097151])
    input_model=image2.cpu().squeeze(0), boxes.to(coco_demo.cpu_device), labels.to(coco_demo.cpu_device), masks.to(coco_demo.cpu_device), scores.to(coco_demo.cpu_device), palette
    result_image1 = combine_masks_tuple(input_model)
    pyplot.imshow(result_image1[:, :, [2, 1, 0]])
    pyplot.show()

    print("testing second image:")


    image = processImage("test2.jpg",800,coco_demo)
    image2 = processImageCPU("test2.jpg",800,coco_demo)

    boxes, labels, masks, scores = loaded(image)
    palette=torch.tensor([3, 32767, 2097151])
    input_model=image2.cpu().squeeze(0), boxes.to(coco_demo.cpu_device), labels.to(coco_demo.cpu_device), masks.to(coco_demo.cpu_device), scores.to(coco_demo.cpu_device), palette
    result_image1 = combine_masks_tuple(input_model)
    pyplot.imshow(result_image1[:, :, [2, 1, 0]])
    pyplot.show()



if __name__ == "__main__":
    main()
