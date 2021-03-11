# Overview

The section provides a brief overview of the functionality provided by
NNHelferlen.
For more details, please visit the API-Section.

## Neural network definitions

The abstract type `DNN` provides signatures to be called as
+ `(m::DNN)(x)`: evaluate x (sample or minibatch)
+ `(m::DNN)(x,y)`: evaluate x and calculate the loss
+ `(m::DNN)(d)`: return the mean loss for a dataset, if d is an iterator
                of type `Knet.Data` or `NNHelferlen.DataLoader`
+ `(m::DNN)((x,y))`: return the mean loss for a x,y-tuple.

Explicit signatures exist for types `Classifier` and `Regressor` with
negative log-likelihood and square loss as loss, respectively.
The type `Chain` wraps a list of layers that are executed sequentially.



## Layer definitions

Several layers are predefined with executable signatures:
+ **MLPs:** different flavours of the simple layer:
        `Dense`: default layer for a vector (i.e. sample)
           or matrix (i.e. mininbatch) as input with logistic
           actvation as default.        
        `Predictions`: Dense layer without activation function.
        `Linear`: TensorFlow-style layer to process high-dimensional
          arrays and identity as default activation.
        `Embed`: embedding layer that adds a first dimension with the
           embeddings to the input.

+ **Convolutional NNs:** to build CNNs `Conv`, `DeConv`, `Pool`
        `UnPool` and `Flat`      
        layers are provided with standard functionality.
        The utilitys include methods for array manipulation, such as
        clipping arrays or adding dimensions.

+ **Others:** additional layers include (please see the API-section for
        a complete list!):
        `Softmax`, `Dropout`, trainable `BatchNorm`, trainable `LayerNorm`.


## Attention Mechanisms

Some attention mechanisms are implemented for use in sequence-to-sequence
networks. If possible projections of values are  precomputed to reduce
computational cost:
+ **AttnBahdanau:** concat- or additive-style attention according to
        Bahdanau, 2015.
+ **AttnLuong:** multiplicative-or general-stype attention according to
        Luong, 2015.
+ **AttnDot:** dot-product-style attention according to
        Luong, 2015.
+ **AttnLocation:** dot-product-style attention according to
        Luong, 2015.
+ **AttnInFeed:** input-feeding attention according to
        Luong, 2015.

A generalised dot-product attention can be computed from
(Query, Key, Value) tuple: `dot_prod_attn(q, k, v)`.

Helpers for transformer networks include functions for positional encoding,
generating padding- and peek-akead-masks and computing
scaled multi-headed attention,
according to Vaswani, 2017.

## Data provider
### Image data

The function `mk_image_minibatch()` can be used to create an
iterator over images, organised in directories, with the first
directory-level as class labels.

Helper functions (such as `image2array()`, `array2image()`, `array2RGB()`)
can be used to transform image data to arrays.
Imagenet-style preprocessing can be achieved with `preproc_imagenet()`,
readable Imagenet class labels of the top predictions are printed by
`predict_imagenet()`.



### DataFrames

Helpers for tabular date include:
+ `dataframe_read`: read a csv-file and return a DataFrame
+ `dataframe_split`: split tabular data in a DataFrame into train and
                validation data; optionally with balancing.
+ `dataframe_minibatches` data provider to turn tabular data from
                a DataFrame (with one sample per row)
                into a Knet-like iterator of minibatches of type `Knet.Data`.

### Texts and NLP

Some utilities are provided for NLP data handling:

+ `WordTokenizer`: a simple tool to encode words as ids.
        The type comes with signatures to en- and decode in both directions.
+ `get_tatoeba_corpus`: download dual-language corpi and provide
        corresponding lists of sentences in two languages.

`seq_minibatch()` and `seq2seq_minibatch()` functions return iterators
to sequences (incl. truncating or padding of the sequences to a common
length). The functions are not restricted to tokenised strings as
sequence elements can be on any type.


## Working with pretrained networks

Layers of pre-trained models can be created from TensorFlow
HDF5-parameter files. It is possible to build a network from
any pretrained TensorFlow model by importing the parameters by
HDF5-constructors for the layers
`Dense`, `Conv`. The flatten-layer `PyFlat` allows for Python-like
row-major-flattening, necessary to make sure, that the parameters
of an imported layer after flattening are in the correct order.



## Training

Although Knet-style is to avoid havyweight interfaces and train networks
with lightweight and flexible optimisers, a train interface
is added that provides TensorBoard logs with online reporting of
minibatch loss, training and validation loss and accuracy.

## Utilities

A number of additional utilities is included. Please have a look at
the utilities section of the API documentation.
