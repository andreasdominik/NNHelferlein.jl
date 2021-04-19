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
