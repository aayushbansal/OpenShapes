# OpenShapes - MATLAB code

We provide MATLAB code for our work on [in-the-wild image synthesis and manipulation](http://www.cs.cmu.edu/~aayushb/OpenShapes/). The beta version of web-app is available [here](openshapes.perception.cs.cmu.edu:5000).

## Demo Video

[![OpenShapes web-app (beta version)!](https://img.youtube.com/vi/Yi8Z5AeBRxI/0.jpg)](https://youtu.be/Yi8Z5AeBRxI)

Click above to see video!

## Usage

Our work heavily builds on [COCO](http://cocodataset.org/#home) and [COCO Panoptic Segmentation dataset](https://github.com/cocodataset/panopticapi). One  would require the images and labels (semantic and panoptic) to run this code. For the sake of reproduction, please use the following to setup dataset.

Please download coco dataset (alongwith semantic and panoptic labels) to generate images using this [dropbox link](https://www.dropbox.com/s/qogqcgkw0yx1akz/coco.tar.gz?dl=0), and place it in *dataset/* directory.

```make
cd dataset
tar -xvzf coco.tar.gz
cd ..
```

Additionally, following command would be required to setup some initial data.

```make
# extract data for faster processing
cd cachedir
tar -xvzf coco.tar.gz
cd ..
```

Once these two things are done, one may just run **OpenShapes** from MATLAB terminal. Please make sure you are in *matlab* directory. The results will be stored in *./cachedir/coco/OpenShapes/* directory.

## Compute

The code does not require GPUs or other expensive hardware. You may run it on your regular machines. It is however suggested to keep data on a ssd or fast drive to get fast results.


## Reference

Please consider citing our work if you find this code useful. 

```make
@inproceedings{OpenShapes,
  author    = {Aayush Bansal and
               Yaser Sheikh and
               Deva Ramanan},
  title     = {Shapes and Context: In-the-wild Image Synthesis & Manipulation},
  booktitle   = {CVPR},
  year      = {2019},
}
```

## Other Datasets

The code can be easily modified for other similar datasets by changing the paths to data and providing dataset specific information in *./experiments/OpenShapes.m* file. Feel free to reach out if you encounter any trouble.



