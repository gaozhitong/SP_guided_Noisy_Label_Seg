# Data Process Guideline
## 1. Download Original Datasets
1. ISIC datasets (training: https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip test: https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
2. JSRT datasets (link / zip) (image: http://db.jsrt.or.jp/eng.php segmentation mask: https://www.isi.uu.nl/Research/Databases/SCR/, we also provided a sorted dataset in the this link)
## 2. Simulated Noise Patterns
Material:
1. data_process/pipeline.py
2. ISIC & JSRT clean data
3. paramters: change params to your directory in __main__()

Function:
1. train/val split  (here val set is used as test set and remain untouched) 
2. noisy label generation
3. convert .png to .json, used as unified input labels.

```angular2html
# change to your parameters first (Params, roots, save_dir)
python pipeline.py
```

## 2*. [Optional] Crop
**Clavicles are particularly small, to facilitate ﬁne-grained segmentation and reduce consuming time, we crop their region of interest by statistics on the training set.
```angular2html
# change to your parameters first (Params, roots, rois)
python crop_pipeline.py
```

## 3. Superpixel Generation
Generate superpixels for input images
```angular2html
# change to your parameters first (subdir, Params, ...)
python superpixel_generation.py
```

## 4. Noise ratio statistics
**We assume noise ratios are given following previous works, thus obtain them here.
```angular2html
# change to your parameters first (subdir, cls, Params, ...)
python stat_niose_ratio.py
```


** Feel free to contact us if confused with above pipeline, and we will make "readme" more clear.