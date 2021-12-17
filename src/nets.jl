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




"""
    struct Chain

Simple wrapper to chain layers and execute them one after another.
"""
struct Chain <: DNN
    layers
    Chain(layers...) = new(layers)
end



function Base.summary(mdl::DNN; indent=0)
    n = get_n_params(mdl)
    ls = length(mdl.layers)
    s1 = "$(typeof(mdl)) with $ls layers,"
    return print_summary_line(indent, s1, n)
end


"""
    function print_network(mdl::DNN)

Print a network summary of any model of Type `DNN`.
If the model has a field `layers`, the summary of all included layers
will be printed recursively.
"""
function print_network(mdl::DNN; n=0, indent=0)

    top = indent == 0
    if top
        println("Neural network summary:")
        println(summary(mdl))
        println("Details:")
    else
        println(summary(mdl, indent=indent))
    end

    indent += 4
    println(" ")
    for l in mdl.layers
        if l isa DNN
            n = print_network(l, n=n, indent=indent)
            println(" ")
        else
            println(summary(l, indent=indent))
            n += 1
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
    VAE(e,d)
Separate predefinded chains (idella, but not necessarily of type `Chain`) 
for encoder and decoder must be specified.

### Signatures: 
    (vae::VAE)(x)
    (vae::VAE)(x,x)
Called with one argument prodict will be executed; 
with two arguments (args x and y should be identical for the autoencoder)
the loss will be returned.    

### Details:
The loss is calculated as the sum of element-wise error squares plus
the Kullback-Leibler-Divergence to adapt the distributions of the
bottleneck codes.

Input and output
of the autoencoder are cropped in a way that their seize is idendical before
loss calculation (and before prediction); i.e. the output has alwas the same dimensions
as the input, even if the last layer generates a different shape
(this is especially of interest for convolutional autoencoders).
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
        ζ = KnetArray(ζ)
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
    if y == nothing
        return x
    else
        n = length(x)
        loss = sum(abs2, x .- y) / n
        loss_KL = -sum(1 .+ logσ² .- μ.*μ .- σ²) / (2n)
        return loss + loss_KL
    end
end