# Basic layer defs
#
# (c) A. Dominik, 2020



"""
    abstract type Layer end

Mother type for layers hierarchy.
"""
abstract type Layer
end


"""
    struct Dense  <: Layer

Default Dense layer.

### Constructors:
+ `Dense(w, b, actf)`: default constructor
+ `Dense(i::Int, j::Int; actf=sigm)`: layer of j neurons with
        i inputs.
+ `Dense(h5::HDF5.File, group::String; trainable=false, actf=sigm)`:
+ `Dense(h5::HDF5.File, kernel::String, bias::String;
        trainable=false, actf=sigm)`: layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.
"""
struct Dense  <: Layer
    w
    b
    actf
    Dense(w, b, actf) = new(w, b, actf)
    Dense(i::Int, j::Int; actf=Knet.sigm) = new(Knet.param(j,i), Knet.param0(j), actf)
 end

Dense(h5::HDF5.File, group::String; trainable=false, actf=Knet.sigm) =
    Dense(h5, "$group/$group/kernel:0","$group/$group/bias:0", trainable=trainable, actf=actf)

function Dense(h5::HDF5.File, kernel::String, bias::String; trainable=false, actf=Knet.sigm)

    w = read(h5, kernel)
    b = read(h5, bias)

    if CUDA.functional()
        w = KnetArray(w)
        b = KnetArray(b)
    end
    if trainable
        w = Param(w)
        b = Param(b)
    end

    (o, i) = size(w)
    println("Generating $actf Dense layer from hdf with $o neurons and $i fan-in.")
    return Dense(w, b, actf)
end

(l::Dense)(x) = l.actf.(l.w * x .+ l.b)

function Base.summary(l::Dense; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Dense layer $i → $o with $(l.actf),"
    return print_summary_line(indent, s1, n)
end


"""
    struct Linear  <: Layer

Almost standard dense layer, but functionality inspired by
the TensorFlow-layer:
+ capable to work with input tensors of
  any number of dimensions
+ default activation function `indetity`
+ optionally without biases.

The shape of the input tensor is preserved; only the size of the
first dim is changed from in to out.

### Constructors:
+ `Linear(i::Int, j::Int; bias=true, actf=identity)` weher `i` is fan-in
        and `j` is fan-out.

### Keyword arguments:
+ `bias=true`: if false biases are fixed to 0.0
+ `actf=identity`: activation function.
"""
struct Linear  <: Layer
    w
    b
    actf
    Linear(w, b, actf) = new(w, b, actf)
    Linear(i::Int, j::Int; bias=true, actf=identity) = new(Knet.param(j,i),
            bias ? Knet.param0(j) : init0(j), actf)
 end

 function (l::Linear)(x)
     j,i = size(l.w)   # get fan-in and out
     siz = vcat(j, collect(size(x)[2:end]))
     x = reshape(x, i,:)
     y = l.actf.(l.w * x .+ l.b)
     return reshape(y, siz...)
 end

function Base.summary(l::Linear; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Linear layer $i → $o, with $(l.actf),"
    return print_summary_line(indent, s1, n)
end




"""
    struct Conv  <: Layer

Default Conv layer.

### Constructors:
+ `Conv(w, b, padding, actf)`: default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu; kwargs...)`: layer with
    o kernels of size (w1,w2) for an input of i layers.
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`:
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`: layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.

### Keyword arguments:
+ `padding=0`: the number of extra zeros implicitly concatenated
        at the start and end of each dimension.
+ `stride=1`: the number of elements to slide to reach the next filtering window.
+ `dilation=1`: dilation factor for each dimension.
+ `...` See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `conv4()` are supported.
"""
struct Conv  <: Layer
    w
    b
    actf
    kwargs
    Conv(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,i,o; init=xavier_normal), Knet.param0(1,1,o,1),
                actf, kwargs)
end

(c::Conv)(x) = c.actf.(Knet.conv4(c.w, x; c.kwargs...) .+ c.b)

Conv(h5::HDF5.File, group::String; trainable=false, actf=Knet.relu) =
    Conv(h5, "$group/$group/kernel:0","$group/$group/bias:0", trainable=trainable, actf=actf)

function Conv(h5::HDF5.File, kernel::String, bias::String; trainable=false, actf=Knet.relu)

    w = read(h5, kernel)
    w = permutedims(w, [4,3,2,1])

    b = read(h5, bias)
    b = reshape(b, 1,1,:,1)

    if CUDA.functional()
        w = KnetArray(w)
        b = KnetArray(b)
    end
    if trainable
        w = Param(w)
        b = Param(b)
    end

    (w1, w2, i, o) = size(w)
    pad = (w1-1)÷2
    println("Generating layer from hdf with kernel ($w1,$w2), $i channels, $o kernels and $pad padding.")

    return Conv(w, b, actf; padding=pad)
end

function Base.summary(l::Conv; indent=0)
    n = get_n_params(l)
    k1,k2,i,o = size(l.w)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "Conv layer $i → $o ($k1,$k2)$kwa with $(l.actf),"
    return print_summary_line(indent, s1, n)
end




"""
    struct Pool <: Layer

Pooling layer.

### Constructors:
+ `Pool(;kwargs...)`: max pooling; without kwargs, 2x2-pooling
        is performed.

### Keyword arguments:
+ `window=2`: pooling window size (same for both directions)
+ `...`: See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `pool` are supported.
"""
struct Pool    <: Layer
    kwargs
    Pool(;kwargs...) = new(kwargs)
end

(l::Pool)(x) = Knet.pool(x; l.kwargs...)


function Base.summary(l::Pool; indent=0)
    n = get_n_params(l)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "Pool layer$kwa,"
    s2 = @sprintf("%10d params", n)
    return print_summary_line(indent, s1, n)
end





"""
    struct DeConv  <: Layer

Default deconvolution layer.

### Constructors:
+ `DeConv(w, b, actf, kwargs...)`: default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, kwargs...)`: layer with
    o kernels of size (w1,w2) for an input of i channels.
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`:
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`: layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.

### Keyword arguments:
+ `padding=0`: the number of extra zeros implicitly concatenated
        at the start and end of each dimension (applied to the output).
+ `stride=1`: the number of elements to slide to reach the next filtering window
        (applied to the output).
+ `...` See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `deconv4()` are supported.

"""
struct DeConv  <: Layer
    w
    b
    actf
    kwargs
    DeConv(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    DeConv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,o,i; init=xavier_normal), Knet.param0(1,1,o,1),
            actf, kwargs)
end

(c::DeConv)(x) = c.actf.(Knet.deconv4(c.w, x; c.kwargs...) .+ c.b)

function Base.summary(l::DeConv; indent=0)
    n = get_n_params(l)
    k1,k2,i,o = size(l.w)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "DeConv layer $o → $i ($k1,$k2)$kwa with $(l.actf),"
    return print_summary_line(indent, s1, n)
end




"""
    struct UnPool <: Layer

Unpooling layer.

### Constructors:
+ `UnPool(;kwargs...)`: user-defined unpooling
"""
struct UnPool <: Layer
    kwargs
    UnPool(;kwargs...) = new(kwargs)
end
(l::UnPool)(x) = Knet.unpool(x; l.kwargs...)

function Base.summary(l::UnPool; indent=0)
    n = get_n_params(l)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "UnPool layer$kwa,"
    return print_summary_line(indent, s1, n)
end




"""
    struct Flat <: Layer

Default flatten layer.

### Constructors:
+ `Flat()`: with no options.
"""
struct Flat <: Layer
end
(l::Flat)(x) = Knet.mat(x)


function Base.summary(l::Flat; indent=0)
    n = get_n_params(l)
    s1 = "Flat layer,"
    return print_summary_line(indent, s1, n)
end




"""
    struct PyFlat <: Layer

Flatten layer with optional Python-stype flattening (row-major).
This layer can be used if pre-trained weight matrices from
tensorflow are applied after the flatten layer.

### Constructors:
+ `PyFlat(; python=true)`: if true, row-major flatten is performed.
"""
struct PyFlat <: Layer
    python
    PyFlat(; python=true) = new(python)
end
(l::PyFlat)(x) = l.python ? Knet.mat(permutedims(x, (3,2,1,4))) : mat(x)

function Base.summary(l::PyFlat; indent=0)
    n = get_n_params(l)
    if l.python
        s1 = "PyFlat layer with row-major (Python) flattening,"
    else
        s1 = "PyFlat layer with column-major (Julia) flattening,"
    end
    return print_summary_line(indent, s1, n)
end



"""
    struct Embed <: Layer

Simple type for an embedding layer to embed a virtual onehot-vector
into a smaller number of neurons by linear combination.
The onehot-vector is virtual, because not the vector, but only
the index of the "one" in the vector has to be provided as Integer value
(or a minibatch of integers).

### Fields:
+ w
+ actf

### Constructors:
+ `Embed(v,d; actf=identity):` with
    vocab size v, embedding depth d and default activation function idendity.

### Signatures:
+ `(l::Embed)(x) = l.actf.(w[:,x])` default
  embedding of input tensor x.
"""
struct Embed <: Layer
    w
    actf
    Embed(i, embed; actf=identity) = new(Knet.param(embed,i), actf)
end

(l::Embed)(x) = l.actf.(l.w[:,x])


function Base.summary(l::Embed; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Embed layer $i → $o, with $(l.actf),"
    return print_summary_line(indent, s1, n)
end



"""
    struct Softmax <: Layer

Simple softmax layer to compute softmax probabilities.

### Constructors:
+ `Softmax()`
"""
struct Softmax <: Layer
end
(l::Softmax)(x) = Knet.softmax(x)

function Base.summary(l::Softmax; indent=0)
    n = get_n_params(l)
    s1 = "Softmax layer,"
    return print_summary_line(indent, s1, n)
end



"""
    struct Dropout <: Layer

Dropout layer.
Implemented with help of Knet's dropout() function that evaluates
AutoGrad.recording() to detect if in training or inprediction.
Dropouts are applied only if prediction.

### Constructors:
+ `Dropout(p)` with the dropout rate *p*.
"""
struct Dropout <: Layer
    p
end
(l::Dropout)(x) = Knet.dropout(x, l.p)

function Base.summary(l::Dropout; indent=0)
    n = get_n_params(l)
    s1 = "Dropout layer with p = $(l.p),"
    return print_summary_line(indent, s1, n)
end





"""
    struct BatchNorm <: Layer

Batchnormalisation layer.
Implemented with help of Knet's batchnorm() function that evaluates
AutoGrad.recording() to detect if in training or in prediction.
In training the moments are updated to record the running averages;
in prediction the moments are applied, but not modified.

In addition, optional trainable factor `a` and bias `b` are applied:

```math
y = a \\cdot \\frac{(x - \\mu)}{(\\sigma + \\epsilon)} + b
```

### Constructors:
+ `Batchnom(; trainable=false, channels=0)` will initialise
        the moments with `Knet.bnmoments()` and
        trainable parameters `a` and `b` only if
        `trainable==true` (in this case, the number of channels must
        be defined - for CNNs this is the number of feature maps).

### Details:
2d, 4d and 5d inputs are supported. Mean and variance are computed over
dimensions (2), (1,2,4) and (1,2,3,5) for 2d, 4d and 5d arrays, respectively.

If `trainable=true` and `channels != 0`, trainable
parameters `a` and `b` will be initialised for each channel.

If `trainable=true` and `channels == 0` (i.e. `Batchnom(trainable=true)`),
the params `a` and `b` are not initialised by the constructor.
Instead,
the number of channels is inferred when the first minibatch is normalised
as:
2d: `size(x)[1]`
4d: `size(x)[3]`
5d: `size(x)[4]`
or `0` otherwise.
"""
mutable struct BatchNorm <: Layer
    trainable
    moments
    params
end

function BatchNorm(; trainable=false, channels=0)
    if trainable
        p = init_bn_params(channels)
    else
        p = nothing
    end
    return BatchNorm(trainable, Knet.bnmoments(), p)
end

function (l::BatchNorm)(x)
    if l.trainable
        if length(l.params) == 0
            l.params = init_bn_params(x)
        end

        return Knet.batchnorm(x, l.moments, l.params)
    else
        return Knet.batchnorm(x, l.moments)
    end
end

function init_bn_params(x)

    if x isa Int
        channels = x
    elseif x isa Array
        dims = size(x)
        if length(dims) in (2, 4, 5)
            channels = dims[end-1]
        else
            channels = 0
        end
    else
        channels = 0
    end
    p = Knet.bnparams(Float32, channels)
    if !(p isa Param)
        p = Param(p)
    end
    return p
end

function Base.summary(l::BatchNorm; indent=0)
    n = get_n_params(l)
    if l.trainable
        s1 = "Trainable BatchNorm layer,"
    else
        s1 = "BatchNorm layer,"
    end
    return print_summary_line(indent, s1, n)
end



"""
    struct LayerNorm  <: Layer

Simple layer normalisation (inspired by TFs LayerNormalization).
Implementation is from Deniz Yuret's answer to feature request
429 (https://github.com/denizyuret/Knet.jl/issues/492).

The layer performs a normalisation within each sample, *not* batchwise.
Normalisation is modified by two trainable parameters `a` and `b`
(variance and mean)
added to every value of the sample vector.

### Constructors:
+ `LayertNorm(depth; eps=1e-6)`:  `depth` is the number
        of activations for one sample of the layer.

### Signatures:
+ `function (l::LayerNorm)(x; dims=1)`: normalise x along the given dimensions.
        The size of the specified dimension must fit with the initialised `depth`.
"""
struct LayerNorm  <: Layer
    a
    b
    ϵ
end

function LayerNorm(depth; eps=1e-6)
        a = param(depth; init=ones)
        b = param(depth; init=zeros)
        LayerNorm(a, b, Float32(eps))
end

function (l::LayerNorm)(x; dims=1)
    μ = mean(x, dims=dims)
    σ = std(x, mean=μ, dims=dims)
    return l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b
end

function Base.summary(l::LayerNorm; indent=0)
    n = get_n_params(l)
    s1 = "Trainable LayerNorm layer,"
    return print_summary_line(indent, s1, n)
end





"""
    struct RSeqTagger <: Layer

One layer RNN sequence classifyer that works with minimatches of (time) series data.
minibatch can be a 2- or 3-dimensional Array.
If 2-d, inputs for one step are in one column and the Array has as
many colums as steps.
If 3-d, the last dimension iterates the samples of the minibatch.

Result is an array matrix with the output of the units of all
steps for all smaples of the minibatch (with model depth as first and samples of the minimatch as last dimension).

### Constructors:
+ `RSeqTagger(n_inputs::Int, n_units::Int; u_type=:lstm, o...)`: with
    number of inputs, number of units and unit type.
    Internally the type `Knet.RNN` is used and all keyword arguments
    of `Knet.RNN` may be provided.
"""
struct RSeqTagger <: Layer
    n_inputs
    n_units
    unit_type
    rnn
    RSeqTagger(n_inputs::Int, n_units::Int; u_type=:lstm, o...) =
            new(n_inputs, n_units, u_type, Knet.RNN(n_inputs, n_units; rnnType=u_type, o...))
end

function (rnn::RSeqTagger)(x)
    n_time_steps = size(x)[2]
    x = reshape(x, rnn.n_inputs, n_time_steps, :)
    x = permutedims(x, (1,3,2))
    x = rnn.rnn(x)
    return permutedims(x, (1,3,2)) # [units, time-steps, samples]
end

function Base.summary(l::RSeqTagger; indent=0)
    n = get_n_params(l)
    s1 = "RSeqTagger layer, $(l.n_inputs) → $(l.n_units) of type $(l.unit_type),"
    return print_summary_line(indent, s1, n)
end



"""
    struct RSeqClassifier <: Layer

One layer RNN sequence classifyer that works with minimatches of (time) series data.
minibatch can be a 2- or 3-dimensional Array.
If 2-d, inputs for one step are in one column and the Array has as
many colums as steps.
If 3-d, the last dimension iterates the samples of the minibatch.

Result is always a 2-d matrix with the output of the units of the last
step in each column and one column per sample of the minibatch.

### Constructors:
+ `RSeqClassifer(n_inputs::Int, n_units::Int; u_type=:lstm, o...)`: with
    number of inputs, number of units and unit type.
    Internally the type `Knet.RNN` is used and all keyword arguments
    of `Knet.RNN` may be provided.
"""
struct RSeqClassifier <: Layer
    n_inputs
    n_units
    unit_type
    rnn
    RSeqClassifier(n_inputs::Int, n_units::Int; u_type=:lstm, o...) =
            new(n_inputs, n_units, u_type, Knet.RNN(n_inputs, n_units; rnnType=u_type, o...))
end



function (rnn::RSeqClassifier)(x)
    n_time_steps = size(x)[2]
    x = reshape(x, rnn.n_inputs, n_time_steps, :)
    x = permutedims(x, (1,3,2))
    x = rnn.rnn(x)
    return x[:,:,end]     # [units, samples]
end

function Base.summary(l::RSeqClassifier; indent=0)
    n = get_n_params(l)
    s1 = "RSeqClassifyer layer, $(l.n_inputs) → $(l.n_units) of type $(l.unit_type),"
    return print_summary_line(indent, s1, n)
end



"""
    function hidden_states(l::<RNN_Type>)

Return the hidden states of one or more layers of an RNN.
`<RNN_Type>` is one of `RSeqClassifier`, `RSeqTagger`,
`Knet.RNN`.
"""
function hidden_states(l::Union{RSeqClassifier, RSeqTagger, Knet.RNN})
    if l isa Union{RSeqClassifier, RSeqTagger}
        return l.rnn.h
    elseif l isa Knet.RNN
        return l.h
    else
        return nothing
    end
end

"""
    function cell_states(l::<RNN_Type>)

Return the cell states of one or more layers of an RNN only if
it is a LSTM.
`<RNN_Type>` is one of `RSeqClassifier`, `RSeqTagger`,
`Knet.RNN`.
"""
function cell_states(l::Union{RSeqClassifier, RSeqTagger, Knet.RNN})
    if l isa Union{RSeqClassifier, RSeqTagger} &&
       l.unit_type == :lstm
        return l.rnn.c
    elseif l isa Knet.RNN && l.mode == 2  # i.e. :lstm
        return l.c
    else
        return nothing
    end
end


# return number of params:
#
function get_n_params(mdl)

    n = 0
    for p in params(mdl)
        n += length(p)
    end
    return n
end
