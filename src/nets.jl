# Chains for NNs:
#
# (c) A. Dominik, 2021


"""
    abstract type DNN end

Mother type for DNN hierarchy with implementation for a chain of layers.

### Signatures:
```Julia
(m::DNN)(x) = (for l in m.layers; x = l(x); end; x)
(m::DNN)(x,y) = m(x,y)
(m::DNN)(d::Knet.Data) = mean( m(x,y) for (x,y) in d)
(m::DNN)(d::Tuple) = mean( m(x,y) for (x,y) in d)
(m::DNN)(d::NNHelferlein.DataLoader) = mean( m(x,y) for (x,y) in d)
```
"""
abstract type DNN
end
(m::DNN)(x) = (for l in m.layers; x = l(x); end; x)
(m::DNN)(x,y) = m(x,y)
(m::DNN)(d::Knet.Data) = mean( m(x,y) for (x,y) in d)
(m::DNN)(d::Tuple) = mean( m(x,y) for (x,y) in d)
(m::DNN)(d::NNHelferlein.DataLoader) = mean( m(x,y) for (x,y) in d)

"""
    struct Classifier <: DNN

Classifyer with nll loss.

### Signatures:
    (m::Classifier)(x,y) = nll(m(x), y)
"""
struct Classifier <: DNN
    layers
    Classifier(layers...) = new(layers)
end
(m::Classifier)(x,y) = Knet.nll(m(x), y)





"""
    struct Regressor

Regression network with square loss.

### Signatures:
    (m::Regression)(x,y) = sum(abs2.( m(x) - y))
"""
struct Regressor <: DNN
    layers
    Regressor(layers...) = new(layers)
end
(m::Regressor)(x,y) = sum(abs2, m(x) .- y)

function Base.summary(mdl::Regressor)
    n = get_n_params(mdl)
    ls = length(mdl.layers)
    s1 = "regressor with $ls layers,"
    return @sprintf("%-50s params: %8d", s1, n)
end



"""
    struct Chain

Simple wrapper to chain layers and execute them one after another.
"""
struct Chain <: DNN
    layers
    Chain(layers...) = new(layers)
end



function Base.summary(mdl::DNN)
    n = get_n_params(mdl)
    ls = length(mdl.layers)
    s1 = "$(typeof(mdl)) with $ls layers,"
    return @sprintf("%-50s params: %8d", s1, n)
end


function print_network(mdl::DNN; indent="")

    if indent == ""
        println("Neural network summary:")
        println(summary(mdl))
        println("Details:")
    else
        println(indent*summary(mdl))
    end

    indent *= "    "
    println(" ")
    for l in mdl.layers
        if l isa DNN
            print_network(l, indent=indent)
            println(" ")
        else
            println(indent*summary(l))
        end
    end
end
