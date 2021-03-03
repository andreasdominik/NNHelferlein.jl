#
# Helpers and definitions for transformer networks:
#



"""
    function positional_encoding_sincos(n_embed, n_seq)

Calculate and return a matrix of size [n_embed, n_seq] of
positional encoding values
following the sin and cos style in the paper
*Vaswani, A. et al.; Attention Is All You Need;
31st Conference on Neural Information Processing Systems (NIPS 2017),
Long Beach, CA, USA, 2017.*
"""
function positional_encoding_sincos(n_embed, n_seq)

    angl = [1/(10000^(2*i/n_embed)) for i in 1:n_embed/2]
    angl = angl * permutedims(1:n_seq)
    sins = sin.(angl)
    coss = cos.(angl)
    pe = vcat(sins, coss)
    return pe
end



"""
    struct PositionalEncoding <: Layer

Positional encoding layer. Only *sincos*-style (according to
Vaswani, et al., NIPS 2017) is implemented.

The layer takes an array of any any number of dimensions (>=2), calculates
the Vaswani-2017-style positional encoding and adds the encoding to each plane
of the array.
"""
struct PositionalEncoding <: Layer
    style
    PositionalEncoding(;style=:sincos) = new(style)
end

function (l::PositionalEncoding)(x)
    # only one style implemented yet:
    if l.stype == sincos
        return x .+= positional_encoding_sincos(size(x)[1], size(x)[2])
    else
        return x
    end
end
