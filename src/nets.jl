# Chains for NNs:
#
# (c) A. Dominik, 2021


"""
    abstract type DNN end

Mother type for DNN hierarchy with implementation for a chain of layers.

### Signatures:
    (m::DNN)(x) = (for l in m.layers; x = l(x); end; x)
    (m::DNN)(d::Knet.Data) = mean( m(x,y) for (x,y) in d)

"""
abstract type DNN end
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
(m::Regressor)(x,y) = sum(abs2, m(x)-y)


"""
    struct Chain

Simple wrapper tu chain layers afer each other.
"""
struct Chain <: DNN
    layers
    Chain(layers...) = new(layers)
end
