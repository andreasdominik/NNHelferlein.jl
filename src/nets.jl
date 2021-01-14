# Chains for NNs:
#
# (c) A. Dominik, 2021


"""
    abstract type NeuNet end

Mother type for DNN hierarchy with implementation for a chain of layers.

### Signatures:
    (n::NeuNet)(x) = (for l in n.layers; x = l(x); end; x)
    (m::NeuNet)(d::Knet.Data) = mean( m(x,y) for (x,y) in d)

"""
abstract type NeuNet end
(n::NeuNet)(x) = (for l in n.layers; x = l(x); end; x)
(m::NeuNet)(d::Knet.Data) = mean( m(x,y) for (x,y) in d)
(m::NeuNet)(d::NNHelferlein.DataLoader) = mean( m(x,y) for (x,y) in d)

"""
    struct Classifier <: NeuNet

Classifyer with nll loss.

### Signatures:
    (m::Classifier)(x,y) = nll(m(x), y)
"""
struct Classifier <: NeuNet
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
struct Regressor <: NeuNet
    layers
    Regressor(layers...) = new(layers)
end
(m::Regressor)(x,y) = sum(abs2, m(x)-y)


"""
    struct Chain

Simple wrapper tu chain layers afer each other.
"""
struct Chain <: NeuNet
    layers
    Chain(layers...) = new(layers)
end
