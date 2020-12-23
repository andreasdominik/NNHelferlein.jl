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
+ `Dense(w, b, actf):` default constructor
+ `Dense(i::Int, j::Int; actf=sigm):` layer of j neurons with
        i inputs.
+ `Dense(hdfo::Dict, group::String; trainable=false, actf=sigm):` layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.
"""
struct Dense  <: Layer
    w
    b
    actf
    Dense(w, b, actf) = new(w, b, actf)
    Dense(i::Int, j::Int; actf=sigm) = new(param(j,i), param0(j), actf)
 end

function Dense(hdfo::Dict, group::String; trainable=false, actf=sigm)

    w = hdfo[group][group]["kernel:0"]
    b = hdfo[group][group]["bias:0"]

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
+ `Conv(w, b, padding, actf):` default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, padding=0):` layer with
    o kernels of size (w1,w2) for an input of i layers.
+ `Conv(hdfo::Dict, group::String; trainable=false, actf=relu):` layer
        imported from a hdf5-file from tensorflow with the
        hdf-object hdfo and the group name group.
"""
struct Conv  <: Layer
    w
    b
    padding
    actf
    Conv(w, b, padding, actf) = new(w, b, padding, actf)
    Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, padding=0) =
            new(param(w1,w2,i,o), param0(1,1,o,1), padding, actf)
end

function Conv(hdfo::Dict, group::String; trainable=false, actf=relu)

    w = hdfo[group][group]["kernel:0"]
    w = permutedims(w, [4,3,2,1])

    b = hdfo[group][group]["bias:0"]
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

(c::Conv)(x) = c.actf.(conv4(c.w, x, padding=c.padding) .+ c.b)


struct Pool <: Layer
    kernel
    Pool(k...) = new(k)
end
(l::Pool)(x) = pool(x, window=l.kernel)


struct Flat <: Layer
end
(l::Flat)(x) = mat(x)

struct PyFlat <: Layer
    python
    PyFlat(; python=false) = new(python)
end
(l::PyFlat)(x) = l.python ? mat(permutedims(x, (3,2,1,4))) : mat(x)
