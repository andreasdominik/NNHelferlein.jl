

# Chains

```@docs
NeuNet
```

```@docs
Classifier
```

```@docs
Regressor
```


# Layers

```@docs
Layer
```


```@docs
Dense
```


```@docs
Conv
```


```@docs
Pool
```


```@docs
Flat
```


```@docs
PyFlat
```


```@docs
Embed
```


```@docs
Predictions
```


# Data providers

```@docs
DataLoader
```


## Tabular data

Tabular data is normally provided in table form (csv, ods)
row-wise, i.e. one sample per row.
The helper functions can read the tables and generate Knet compatible
iterators of minibatches.

```@docs
dataframe_read
```


```@docs
dataframe_minibatches
```


```@docs
dataframe_split
```

## Image data

Images as data should be provided in directories with the directory names
denoting the class labels.
The helpers read from the root of a directory tree in which the
first level of sub-dirs tell the class label. All images in the
tree under a class label are read as instances of the respective class.
The following tree will generate the classes `daisy`, `rose` and `tulip`:

```
image_dir/
├── daisy
│   ├── 01
│   │   ├── 01
│   │   ├── 02
│   │   └── 03
│   ├── 02
│   │   ├── 01
│   │   └── 02
│   └── others
├── rose
│   ├── big
│   └── small
└── tulip
```

```@docs
ImageLoader
```


```@docs
mk_image_minibatch
```

## Utilities

```@docs
get_class_labels
```


### Training

```@docs
tb_train!
```

### Evaluation


### ImageNet tools
