API doc of all exported functions are listed here:

# Chains

```@docs
DNN
add_layer!
print_network
```

```@docs
Classifier
Regressor
Chain
VAE
```

# Layers

```@docs
Layer
```

## Fully connected layers

```@docs
Dense
Linear
Embed
```

## Convolutional

```@docs
Conv
DeConv
Pool
UnPool
```

## Recurrent

```@docs
RecurrentUnit
Recurrent
get_hidden_states
get_cell_states
set_hidden_states!
set_cell_states!
reset_hidden_states!
reset_cell_states!
```

## Others

```@docs
Flat
PyFlat
Softmax
Dropout
BatchNorm
LayerNorm
```


## Attention Mechanisms

```@docs
AttentionMechanism
AttnBahdanau
AttnLuong
AttnDot
AttnLocation
AttnInFeed
```


## Layers and helpers for transformers

```@docs
PositionalEncoding
mk_padding_mask
mk_peek_ahead_mask
dot_prod_attn
MultiHeadAttn
separate_heads
merge_heads
```

# Data providers

```@docs
DataLoader
PartialIterator
split_minibatches
```

## Tabular data

Tabular data is normally provided in table form (csv, ods)
row-wise, i.e. one sample per row.
The helper functions can read the tables and generate Knet compatible
iterators of minibatches.

```@docs
dataframe_read
dataframe_minibatches
dataframe_split
mk_class_ids
MBNoiser
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
mk_image_minibatch
get_class_labels
image2array
array2image
array2RGB
```

## Text data

```@docs
WordTokenizer
get_tatoeba_corpus
sequence_minibatch
pad_sequence
truncate_sequence
```



# Training

```@docs
tb_train!
```

# Evaluation

```@docs
predict
predict_top5
hamming_dist
peak_finder_acc
```

# ImageNet tools

```@docs
preproc_imagenet
predict_imagenet
get_imagenet_classes
```


# Other utils

## Utils for transformers

```@docs
positional_encoding_sincos
mk_padding_mask
mk_peek_ahead_mask
dot_prod_attn
```


## Utils for array manipulation

```@docs
crop_array
blowup_array
recycle_array
de_embed
```

## Utils for fixing types in GPU context

```@docs
init0
convert2KnetArray
emptyKnetArray
```
