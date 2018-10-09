##  Models:
| name | caffemodel | prototxt | training set
| --- | --- | --- | --- |
| CurriculumNet(WebVision)  | [inception_v2_webvision.caffemodel](https://sai-pub.s3.cn-north-1.amazonaws.com.cn/malong-research/curriculumnet/inception_v2_webvision.caffemodel)  | [inception_v2_deploy_webvision.prototxt](inception_v2_deploy_webvision.prototxt)|[WebVision 1.0](https://www.vision.ee.ethz.ch/webvision/2017/download.html)
| CurriculumNet(WebVision+ImageNet) | [inception_v2_web_img.caffemodel](https://sai-pub.s3.cn-north-1.amazonaws.com.cn/malong-research/curriculumnet/inception_v2_web_img.caffemodel) | [inception_v2_deploy_web_img.prototxt](inception_v2_deploy_web_img.prototxt) | [WebVision 1.0](https://www.vision.ee.ethz.ch/webvision/2017/download.html) + [ImageNet](http://image-net.org/download)

These models are trained using the strategy described in the [CurriculumNet paper](https://arxiv.org/abs/1808.01097).

##  Performance:

The performance on the validation sets of ImageNet and WebVision are as follows:

| Name | WebVision (Top-1) | WebVision (Top-5) |  ImageNet (Top-1) |  ImageNet (Top-5)
| --- | --- | --- | --- | --- |
| CurriculumNet(WebVision) | 27.9  | 10.8 | 35.2 | 15.1
| CurriculumNet(WebVision+ImageNet) | 24.7 | 8.5 | 24.8 | 7.1

##  Framework:

The deep learning framework is a fork of Caffe with OpenMPI-based Multi-GPU: https://github.com/yjxiong/caffe/

## License

The models are released for non-commercial use. For commercial use, please contact bd@malong.com