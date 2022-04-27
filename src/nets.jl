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

Classifier with nll loss.

### Signatures:
    (m::Classifier)(x,y) = nll(m(x), y)
"""
struct Classifier <: DNN
    layers
    Classifier(layers...) = new(Any[layers...])
end
(m::Classifier)(x,y) = Knet.nll(m(x), y)





"""
    struct Regressor

Regression network with square loss as loss function.

### Signatures:
    (m::Regression)(x,y) = sum(abs2, Array(m(x)) - y)
"""
struct Regressor <: DNN
    layers
    Regressor(layers...) = new(Any[layers...])
end
(m::Regressor)(x,y) = sum(abs2, ifgpu(y) .- m(x))




"""
    struct Chain

Simple wrapper to chain layers and execute them one after another.
"""
struct Chain <: DNN
    layers
    Chain(layers...) = new(Any[layers...])
end

# sequencial interface:
#
import Base: push!, length
push!(n::NNHelferlein.DNN, l) = push!(n.layers, l)
length(n::NNHelferlein.DNN) = length(n.layers)

"""
    add_layer!(n::NNHelferlein.DNN, l)

Add a layer `l` or a chain to a model `n`. The layer is always added 
at the end of the chains. 
The modified model is returned.
"""
function add_layer!(n::NNHelferlein.DNN, l)
    push!(n.layers, l)
    return n
end


import Base.+

"""
    function +(n::DNN, l::Union{Layer, Chain})

The `plus`-operator is overloaded to be able to add layers and chains 
to a network.

### Example:

```Julia
julia> mdl = Classifier() + Dense(2,5)
julia> print_network(mdl)

NNHelferlein neural network summary:
Classifier with 1 layers,                                           15 params
Details:
 
    Dense layer 2 → 5 with sigm,                                    15 params
 
Total number of layers: 1
Total number of parameters: 15


julia> mdl = mdl + Dense(5,5) + Dense(5,1, actf=identity)
julia> print_network(mdl)

NNHelferlein neural network summary:
Classifier with 3 layers,                                           51 params
Details:
 
    Dense layer 2 → 5 with sigm,                                    15 params
    Dense layer 5 → 5 with sigm,                                    30 params
    Dense layer 5 → 1 with identity,                                 6 params
 
Total number of layers: 3
Total number of parameters: 51
```
"""
function +(n::NNHelferlein.DNN, l::Union{NNHelferlein.Layer, NNHelferlein.Chain})
    add_layer!(n, l)
    return n
end




function Base.summary(mdl::DNN; indent=0)
    n = get_n_params(mdl)
    if hasproperty(mdl, :layers)
        ls = length(mdl.layers)
        s1 = "$(typeof(mdl)) with $ls layers,"
    else
        s1 = "$(typeof(mdl)),"
    end
    return print_summary_line(indent, s1, n)
end


"""
    function print_network(mdl::DNN)

Print a network summary of any model of Type `DNN`.
If the model has a field `layers`, the summary of all included layers
will be printed recursively.
"""
function print_network(mdl; n=0, indent=0)

    top = indent == 0
    if top
        println("NNHelferlein neural network summary:")
        println(summary(mdl))
        println("Details:")
    else
        println(summary(mdl, indent=indent))
    end

    indent += 4
    println(" ")
    for pn in propertynames(mdl)
        p = getproperty(mdl, pn)

        if pn == :layers
            for l in p
                if l isa DNN
                    n = print_network(l, n=n, indent=indent)
                    println(" ")
                else
                    println(summary(l, indent=indent))
                    n += 1
                end
            end
        elseif p isa DNN
            n = print_network(p, n=n, indent=indent)
            println(" ")
        elseif p isa Layer
            println(summary(p, indent=indent))
            n += 1
        elseif p isa AbstractArray
            for l in p
                n = print_network(l, n=n, indent=indent)
                #println(summary(l, indent=indent))
                n += 1
            end
        end
    end
    
        if top
            println(" ")
            println("Total number of layers: $n")
            println("Total number of parameters: $(get_n_params(mdl))")
        end
    return n
end

function print_summary_line(indent, line, params)

    LIN_LEN = 60
    s1 = " "^indent * line
    len = length(s1)
    gap = " "
    if len < LIN_LEN
        gap = " "^(LIN_LEN-len)
    end

    s2 = @sprintf("%8d params", params)

    return "$s1 $gap $s2"
end



"""
    struct VAE

Type for a generic variational autoencoder.

### Constructor:
    VAE(encoder, decoder)
Separate predefind chains (ideally, but not necessarily of type `Chain`) 
for encoder and decoder must be specified.
The VAE needs the 2 parameters mean and variance to define the distribution of each
code-neuron in the bottleneck-layer. In consequence the encoder output must be 2 times 
the size of the decoder input
(in case of dense layers: if encoder output is a 8-value vector,
4 codes are defined and the decoder input is a 4-value vector;
in case of convolutional layers the number of encoder output channels
must be 2 times the number of the encoder input channels - see the examples). 

### Signatures: 
    (vae::VAE)(x)
    (vae::VAE)(x,y)
Called with one argument, predict will be executed; 
with two arguments (args x and y should be identical for the autoencoder)
the loss will be returned.    

### Details:
The loss is calculated as the sum of element-wise error squares plus
the *Kullback-Leibler-Divergence* to adapt the distributions of the
bottleneck codes:
```math
\\mathcal{L} = \\frac{1}{2} \\sum_{i=1}^{n_{outputs}} (t_{i}-o_{i})^{2} - 
               \\frac{1}{2} \\sum_{j=1}^{n_{codes}}(1 + ln\\sigma_{c_j}^{2}-\\mu_{c_j}^{2}-\\sigma_{c_j}^{2}) 
```

Output
of the autoencoder is cropped to the size of input before
loss calculation (and before prediction); i.e. the output has always the same dimensions
as the input, even if the last layer generates a bigger shape.
"""
struct VAE <: DNN
    layers
    VAE(e,d) = new([e,d])
end

function (vae::VAE)(x, y=nothing)
    
    # encode and
    # calc size of decoder input (1/2 of encoder output):
    #
    size_in = size(x)
    x = vae.layers[1](x)
    size_dec_in = [size(x)...]
    size_dec_in[end-1] = size_dec_in[end-1] ÷ 2
    
    # separate μ and σ:
    #
    x = mat(x)
    code_size = size(x)
    n_codes = code_size[1] ÷ 2

    μ = x[1:n_codes,:]
    logσ² = x[n_codes+1:end,:]
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)
    
    # variate:
    #
    ζ = randn(Float32, size(μ))
    if CUDA.functional()
        ζ = convert2KnetArray(ζ)
    end
    
    x = μ .+ ζ .* σ
    
    # reshape codes to fit encoder input
    # and decode:
    #
    x = reshape(x, size_dec_in...)
    x = vae.layers[2](x)
    x = crop_array(x, size_in)
       
    # calc loss, if y given:
    #
    if isnothing(y)
        return x
    else
        n = length(x)
        loss = sum(abs2, x .- y) / 2
        loss_KL = -sum(1 .+ logσ² .- abs2.(μ) .- σ²) / 2
        return loss + loss_KL
    end
end
