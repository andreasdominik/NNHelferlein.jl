# NNHelferlein - ADo's Neural Networks Little Helpers

The package provides helpers and utilities mainly to be
used with the Knet package to build artificial neural networks.

The package follows mainly the Knet-style; i.e. all networks can be
trained with the Knet-iterators, all layers can be used together with
Knet-style *quickly-self-written* layers, all Knet-networks can be trained
with tb_train(), all data providers can be used together, ...


## Installation

For installation please refer to the README @github:
<https://github.com/andreasdominik/NNHelferlein.jl>

## First Steps

NNHelferlein provides quick and easy definition, training and
validation of neural network chains.

### Symbolic API
The Keras-like symbolic API allows for building simple Chains,
Classifiers and Regressors from predefined or self-written 
layers or functions.

A first example may be the famous MNIST handwriting recognition 
data. Let us assume the data is already loaded in minibatches 
in a `dtrn` iterator and a MLP shall do the job. 
The remainder is as little as:

```julia
mlp = Classifier(Dense(784, 256),
                 Dense(256, 64), 
                 Dense(64, 10, actf=identity)))


mlp = tb_train!(mlp, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

Chains may be built of type `Chain`, `Classifier` or `Regressor`.
Simple `Chain`s bring only a signature `model(x)` to compute 
forward computations
of a data-sample, a minibatch of data as well as many minibatches of data
(the dataset -here: dtrn- must be an iterable object that provides
one minibatch on every call).

`Classifier`s and `Regressor`s in addition already come with signatures
for loss calculation of (x,y)-minibatches (`model(x,y)`) 
with crossentropy loss
(i.e. negative log-likelihood) and square-loss respectively. This is why 
for both types the last layer must not have an activation function
(the *Helferlein* `Dense`-layer comes with a logistic/sigmoid activation
by default; alternatively the `Linear`-layer can be used that have 
no default activation function).

The function `tb_train!()`
updates the model with the possibility to specify optimiser, training
and validation data or an optional split ratio to perform a random 
training/validation split. The function offers a multitude of 
other options (see the API-documentation for details) and writes
tensorboard log-files that allow for online plotting of the 
training progress during training via tensorboard.

A second way to define a model is the `add_layer!()`-syntax, here shown
for a simple LeNet-like model for the same problem:

```julia
lenet = Classifier()

add_layer!(lenet, Conv(5,5,1,20))
add_layer!(lenet, Pool())
add_layer!(lenet, Conv(5,5,20,50))
add_layer!(lenet, Pool())
add_layer!(lenet, Flat())
add_layer!(lenet, Dense(800,512))
add_layer!(lenet, Dense(512,10, actf=identity))

mlp = tb_train!(lenet, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

Of course, both possibilities can be combined as desired; the
following code gives a similar model:

```julia
filters = Chain(Conv(5,5,1,20),
                Pool(),
                Conv(5,5,20,50),
                Pool())
classif = Chain(Dense(800,512),
                Dense(512,10, actf=identity))

lenet2 = Classifier(filters, 
                   Flat())
add_layer!(lenet2, classif)

mlp = tb_train!(lenet2, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

Models can be summarised with the `print_network()`-helper:

```julia
julia> print_network(lenet)
Neural network summary:
Classifier with 7 layers,                                       440812 params
Details:
 
    Conv layer 1 → 20 (5,5) with relu,                             520 params
    Pool layer,                                                      0 params
    Conv layer 20 → 50 (5,5) with relu,                          25050 params
    Pool layer,                                                      0 params
    Flat layer,                                                      0 params
    Dense layer 800 → 512 with sigm,                            410112 params
    Dense layer 512 → 10 with identity,                           5130 params
 
Total number of layers: 7
Total number of parameters: 440812
```

### Free model definition
Another way of model definition gives the full freedom 
to define a forward function as pure Julia code. 
In the Python world this type of definition is often referred to  
as the functional API - in the Julia world we hesitate calling 
it an API, 
because at the end of the day all is just out-of-the-box Julia!
Each model just needs a type able to store all parameters, 
a signature `model(x)` to compute a forward run and predict
the result and a signature `model(x,y)` to calculate the loss.

For the predefined `Classifier` and `Regressor` types the signatures are 
predefined - for own models, this can be easily done in a few lines of
code.

The LeNet-like example network for MNIST may be written as:

#### The type and constructor:
```julia
struct LeNet
    drop1
    conv1
    pool1
    conv2
    pool2
    flat
    drop2
    mlp
    predict
    function LeNet(;drop=0.2)
        return new(Dropout(drop),
                   Conv(5,5,1,20),
                   Pool(),
                   Conv(5,5,20,50),
                   Pool(),
                   Flatten(),
                   Dropout(drop)
                   Dense(800, 512),
                   Dense(512, 10, actf=identity))
end
```
Of course the model may be configured with by giving the constructor
more parameters.
Also the code may be written better organised by combining
layers to `Chains`.


#### The forward signature:
```julia
function (nn::LeNet)(x)
    x = drop1(x)
    x = conv1(x)
    x = pool1(x)
    x = conv2(x)
    x = pool2(x)
    x = flat(x)
    x = drop2(x)
    x = mlp(x)
    x = predict(x)
    return x
end
```

#### The loss-signature:
```julia
function (nn::LeNet)(x,y)
    return nll(nn(x), y)
end
```

Here we use the `Knet.nll()` function to calculate the crossentropy. 

That's it!

Belive it or not - that's all you need to leave the 
limitations of the Python world behind and playfully design any 
innovative neural network in just a couple of lines of Julia code.

The next step is to have a look at the examples
in the GitHub repo:

```@contents
Pages = [
    "examples.md"
    ]
Depth = 2
```

## Overview

```@contents
Pages = [
    "overview.md"
    ]
Depth = 2
```



## API Reference

```@contents
Pages = [
    "api.md"
    ]
Depth = 2
```

## Index

```@index
```
