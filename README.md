# Canonical-Sg2Im

#### [Roei Herzig*](https://roeiherz.github.io/), [Amir Bar*](http://www.amirbar.net/), [Huijuan Xu](https://cs-people.bu.edu/hxu/), [Gal Chechik](https://chechiklab.biu.ac.il/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Amir Globerson](https://www.cs.tau.ac.il/~gamir/)

Main project [page](https://roeiherz.github.io/CanonicalSg2Im/).

**Code and models will be released soon.**

<!---[alt text](Figures/teaser.png)--->
<img src="Figures/teaser.png" width="1000">

Generation of scenes with many objects. Our method achieves better performance on such scenes than previous methods. Left: A partial input scene graph.
Middle: Generation using [1]. Right: Generation using our proposed method.


### Our novel contributions are:
1. We propose a model that uses canonical representations of SGs, thus obtaining stronger invariance properties. This in turn leads to **generalization on semantically equivalent graphs and improved robustness to graph size and noise** in comparison to existing methods.
2. We show how to learn the canonicalization process from data.
3. We use our canonical representations within an SG-to-image model and demonstrate our approach results in an **improved generation** on Visual Genome, COCO, and CLEVR, compared to the state-of-the-art baselines.


## Setup
To install the environment, following the instructions:
- `install` [conda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
- `conda create --name ag2vid`
- `conda activate ag2vid`
- `conda install ipython`
- `pip install ninja yacs cython matplotlib tqdm opencv-python`
- `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
- `pip install pillow tensorboardX pandas joblib`
- `pip install h5py imageio lmdb`
- `pip install scipy==1.3.3`


## Dependencies
To get started with the framework, install the following dependencies:
- [Python 3.7](https://www.python.org/)
- [Pytorch 1.1.0](https://www.https://pytorch.org/)
- [matplotlib 3.1.3](http://matplotlib.org/)
- [h5py 2.9.0](http://www.h5py.org/)
- [numpy 1.18.1](http://www.numpy.org/)
- [scipy 1.3.3]()
- [pandas 1.0.3]()
- [imageio 2.8.0]()
- [tensorboardX 1.8]()
- [opencv-python 4.2.9]()
- [joblib 0.14.1]()
- [lmdb 0.98]()

<!--Run `"pip install -r requirements.txt"`  - to install all the requirements.-->


## Usage
<!---
1. Run `"python Run.py download"` to download and extract train, validation and test data. The data already contains the result of applying the baseline detecor over the VisualGenome data. 
2. Run `"python Run.py eval gpi_linguistic_pretrained <gpu-number>"` to evaluate the pre-trained model of our best variant, linguistic with multi-head attention. (recall@100 SG Classification).
3. Run `"python Run.py train gpi_linguistic <gpu-number>"` to train a new model (linguistic with multi-head attention).
4. Run `"python Run.py eval gpi_linguistic_best <gpu-number>"` to evaluate the new model. (recall@100 SG Classification).
--->


## Notes

**Please note that the main part of the code has been released, though we are still testing it to fix possible glitches. Thank you.**

This implementation is built on top of https://github.com/google/sg2im.



## References
[1] Justin Johnson, Agrim Gupta, Li Fei-Fei, [Image Generation from Scene Graphs](https://arxiv.org/abs/1804.01622), 2018.


## Citation

```
@inproceedings{herzig2019canonical,
 author    = {Herzig, Roei and Bar, Amir and Xu, Huijuan and Chechik, Gal and Darrell, Trevor and Globerson, Amir},
 title     = {Learning Canonical Representations for Scene Graph to Image Generation},
 booktitle = {Proc. of the European Conf. on Computer Vision (ECCV)},
 year      = {2020}
}
```


