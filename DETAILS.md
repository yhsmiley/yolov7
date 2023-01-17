## Exponential Moving Average

This is a special case of SWA: https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/

## Gradient Accumulation

- to overcome the problem of batch size being limited by GPU memory
- Gradient accumulation means running a configured number of steps without updating the model variables while accumulating the gradients of those steps and then using the accumulated gradients to compute the variable updates.

## Image Weights

Samples images from the training set weighted by their inverse mAP from the previous epoch's testing.
Attempt to load images more frequently that contain more classes with worse metrics from the previous epoch.

## Auto Anchor

- AutoAnchor runs before training to ensure the anchors (if supplied) are a good fit for the data. If they are not, then new anchors are computed and evolved and attached to the model automatically.
- learns anchor boxes based on the distribution of bounding boxes in the custom dataset with K-means and genetic learning algorithms
- anchors are parameters of the Detect() layer
- to view anchors:
    - m = model.model[-1]  # Detect() layer
    - print(m.anchor_grid.squeeze())  # print anchors

## Fusion of Conv and BN

- https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3
- https://towardsdatascience.com/speed-up-inference-with-batch-normalization-folding-8a45a83a89d8

## Quad Dataloader

https://github.com/ultralytics/yolov5/issues/1898

- allow some benefits of higher --img size training at lower --img sizes.
- This quad-collate function will reshape a batch from 16x3x640x640 to 4x3x1280x1280, which does not have much effect by itself as it is only rearranging the mosaics in the batch, but which interestingly allows for 2x upscaling of some images within the batch (one of the 4 mosaics in each quad is upscaled by 2x, the other 3 mosaics are deleted).
- consider --img 640 --quad as a middle ground that trains with the speed of --img 640 but with a bit of the higher mAP seen when training directly at --img 1280 (which of course would normally take 4x longer than training at --img 640).

## Label Smoothing

While label smoothing improves the robustness, it will cause the performance growth to slow down in the early stage of training, and will lose in class information. It also needs to adjust the degree of smoothing. Different values need to be adjusted for different data sets to obtain better results, and the growth effect is not great.

## Hyperparameters

In general, increasing augmentation hyperparameters will reduce and delay overfitting, allowing for longer trainings and higher final mAP. Reduction in loss component gain hyperparameters like hyp['obj'] will help reduce overfitting in those specific loss components. For an automated method of optimizing these hyperparameters, see [Hyperparameter Evolution Tutorial](https://github.com/ultralytics/yolov5/issues/607).
