# Basic layer defs
#
# (c) A. Dominik, 2020



"""
    abstract type Layer end

Mother type for layers hierarchy.
"""
abstract type Layer end


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





"""
    struct Conv  <: Layer

Default Conv layer.

### Constructors:
+ `Conv(w, b, padding, actf)`: default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, padding=0)`: layer with
    o kernels of size (w1,w2) for an input of i layers.
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`:
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`: layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.
"""
struct Conv  <: Layer
    w
    b
    padding
    actf
    Conv(w, b, padding, actf) = new(w, b, padding, actf)
    Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, padding=0) =
            new(Knet.param(w1,w2,i,o; init=xavier_normal), Knet.param0(1,1,o,1), (padding,padding), actf)
end

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
    pad = ((w1-1)÷2, (w2-1)÷2)
    println("Generating layer from hdf with kernel ($w1,$w2), $i channels, $o kernels and $pad padding.")

    return Conv(w, b, pad, actf)
end

(c::Conv)(x) = c.actf.(Knet.conv4(c.w, x, padding=c.padding) .+ c.b)



"""
    struct Pool <: Layer

Pooling layer.

### Constructors:
+ `Pool(;kwargs...)`: user-defined pooling; without kwargs, 2x2-pooling
        is performed.

### Keyword arguments:
See the Knet documentation for Details:
https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
"""

struct Pool
    kwargs
    Pool(;kwargs...) = new(kwargs)
end
(l::Pool)(x) = Knet.pool(x; l.kwargs...)




"""
    struct DeConv  <: Layer

Default deconvolution layer.

### Constructors:
+ `DeConv(w, b, actf, kwargs...)`: default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, kwargs...)`: layer with
    o kernels of size (w1,w2) for an input of i layers.
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`:
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu)`: layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.
"""
struct DeConv  <: Layer
    w
    b
    actf
    kwargs
    DeConv(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    DeConv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,i,o; init=xavier_normal), Knet.param0(1,1,o,1),
            actf, kwargs)
end

(c::DeConv)(x) = c.actf.(Knet.deconv4(c.w, x; kwargs...) .+ c.b)



"""
    struct UnPool <: Layer

Unpooling layer.

### Constructors:
+ `UnPool()`: default 2×2 unpooling
+ `UnPool(k...)`: user-defined unpooling
"""
struct UnPool <: Layer
    kwargs
    UnPool(;kwargs...) = new(kwargs)
end
(l::UnPool)(x) = Knet.unpool(x; l.kwargs...)


"""
    struct Flat <: Layer

Default flatten layer.

### Constructors:
+ `Flat()`: with no options.
"""
struct Flat <: Layer
end
(l::Flat)(x) = Knet.mat(x)



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
+ `Embed(i,j; actf=identity) = new(param(j,i), actf):` with
    input size i, output size j and default activation function idendity.

### Signatures:
+ `(l::Embed)(x) = l.actf.(w[:,x])` default
  embedding of input vector x.
"""
struct Embed <: Layer
    w
    actf
    Embed(i, embed; actf=identity) = new(Knet.param(embed,i), actf)
end

(l::Embed)(x) = l.actf.(l.w[:,x])

"""
    struct Predictions <: Layer

Simple wrapper around a Dense layer without activation function
that can used as output layer (because all loss-functions
of the package assume raw output activations).

### Constructors:
+ `Predictions(i::Int, j:Int)`: with
    input size i, output size j activation function idendity.
+ `Predictions(h5::HDF5.File, group::String; trainable=false)`:
+ `Predictions(h5::HDF5.File, kernel::String, bias::String;
               trainable=false)`: with
    an hdf5-object and group name of the output layer.
"""
struct Predictions <: Layer
    Predictions(i::Int,j::Int) = Dense(i, j, actf=identity)
    Predictions(h5::HDF5.File, group::String; trainable=false) =
                Predictions(h5, "$group/$group/kernel:0","$group/$group/bias:0",
                            trainable=trainable)
    Predictions(h5::HDF5.File, kernel::String, bias::String;
                trainable=false) =
                Dense(h5, kernel, bias, trainable=trainable, actf=identity)
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



"""
    struct BatchNorm <: Layer

Batchnormalisation layer.
Implemented with help of Knet's batchnorm() function that evaluates
AutoGrad.recording() to detect if in training or in prediction.
In training the moments are updated to record the running averages;
in prediction the moments are applied, but not modified.

### Constructors:
+ `Batchnom()` will initialise the moments with `Knet.bnmoments()`.
"""
struct BatchNorm <: Layer
    moments
    BatchNorm() = new(Knet.bnmoments())
end
(l::BatchNorm)(x) = Knet.batchnorm(x, l.moments)



struct RSeqTaggr
    n_inputs
    n_units
    unit_type
    rnn
    RSeqTaggr(n_inputs::Int, n_units::Int; u_type=:lstm) =
            new(n_inputs, n_units, u_type, Knet.RNN(n_inputs, n_units, rnnType=u_type))
end

function (rnn::RSeqTaggr)(x)
    n_time_steps = size(x)[2]
    x = reshape(x, rnn.n_inputs, n_time_steps, :)
    x = permutedims(x, (1,3,2))
    x = rnn.rnn(x)
    return permutedims(x, (1,3,2)) # [units, time-steps, samples]
end



"""
    struct RSeqClassifyr <: Layer

One layer RNN sequence classifyer that works with minimatches of (time) series data.
minibatch can be a 2- or 3-dimensional Array.
If 2-d, inputs for one step are in one column and the Array has as
many colums as steps.
If 3-d, the last dimension iterates the samples of the minibatch.

Result is always a 2-d matrix with the output of the units of the last
step in each column and one column per sample of the minibatch.

### Constructors:
+ `RSeqClassifyr(n_inputs::Int, n_units::Int; u_type=:lstm)`: with
    number of inputs, number of units and unit type.

            new(n_inputs, n_units, u_type, Knet.RNN(n_inputs, n_units, rnnType=u_type))
"""
struct RSeqClassifyr
    n_inputs
    n_units
    unit_type
    rnn
    RSeqClassifyr(n_inputs::Int, n_units::Int; u_type=:lstm) =
            new(n_inputs, n_units, u_type, Knet.RNN(n_inputs, n_units, rnnType=u_type))
end



function (rnn::RSeqClassifyr)(x)
    n_time_steps = size(x)[2]
    x = reshape(x, rnn.n_inputs, n_time_steps, :)
    x = permutedims(x, (1,3,2))
    x = rnn.rnn(x)
    return x[:,:,end]     # [units, samples]
end
