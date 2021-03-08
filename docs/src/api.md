

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

```@docs
Chain
```

# Layers

```@docs
Layer
```

## Fully connected layers

```@docs
Dense
```

```@docs
Linear
```

```@docs
Embed
```

```@docs
Predictions
```

## Convolutional

```@docs
Conv
```

```@docs
DeConv
```

```@docs
Pool
```

```@docs
UnPool
```

## Recurrent

```@docs
RSeqClassifier
```

## Others

```@docs
Flat
```

```@docs
PyFlat
```


```@docs
Softmax
```


```@docs
Dropout
```

```@docs
BatchNorm
```

```@docs
LayerNorm
```


## Attention Mechanisms

```@docs
AttentionMechanism
```

```@docs
AttnBahdanau
```

```@docs
AttnLuong
```

```@docs
AttnDot
```

```@docs
AttnLocation
```

```@docs
AttnInFeed
```


## Layers for transformers

```@docs
PositionalEncoding
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

```@docs
get_class_labels
```

```@docs
image2array
```

```@docs
array2image
```

```@docs
array2RGB
```

## Text data

```@docs
WordTokenizer
```



# Training

```@docs
tb_train!
```

# Evaluation

```@docs
predict
```

```@docs
predict_top5
```

# ImageNet tools

```@docs
preproc_imagenet
```
```@docs
predict_imagenet
```


# Other utils

## Utils for transformers

```@docs
positional_encoding_sincos
```

```@docs
mk_padding_mask
```

```@docs
mk_peek_ahead_mask
```

```@docs
dot_prod_attn
```


## Utils for array manipulation

```@docs
crop_array
```

```@docs
blowup_array
```

```@docs
recycle_array
```

## Utils for fixing types in GPU context

```@docs
init0
```

```@docs
convert2KnetArray
```
