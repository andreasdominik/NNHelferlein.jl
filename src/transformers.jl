#
# Helpers and definitions for transformer networks:
#



"""
    function positional_encoding_sincos(n_embed, n_seq)

Calculate and return a matrix of size `[n_embed, n_seq]` of
positional encoding values
following the sin and cos style in the paper
*Vaswani, A. et al.; Attention Is All You Need;
31st Conference on Neural Information Processing Systems (NIPS 2017),
Long Beach, CA, USA, 2017.*
"""
function positional_encoding_sincos(n_embed, n_seq)

    angl = [1/(10000^(2*i/n_embed)) for i in 1:n_embed/2]
    angl = angl * permutedims(1:n_seq)
    pos_enc = vcat(sin.(angl), cos.(angl))
    return convert2KnetArray(pos_enc)
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
    # if l.style == sincos
        x = convert2KnetArray(x) .+ positional_encoding_sincos(size(x)[1], size(x)[2])
        return x
end



"""
    function mk_padding_mask(x; pad=3)

Make a padding mask; i.e. return an Array of type
`KnetArray{Float32}` (or `Array{Float32}`) similar to `x` but with
two additional dimension of size 1 in teh middle (this will represent the
2nd seq_len and the number of heads) in multi-head attention
and the
value `1.0` at each position where `x` is `pad` and `0.0` otherwise.

The function can be used for creating padding masks for attention
mechanisms.
"""
function mk_padding_mask(x; pad=3)

    return reshape(convert2KnetArray(x .== pad), size(x)[1],1,1,size(x)[2])
end


"""
    function mk_peek_ahead_mask(x; dim=1)

Return a matrix of size `[n_seq, n_seq]` filled with 1.0 and the *uppper triangle*
set to 0.0.
Type is `KnetArray{Float32}` in GPU context, `Array{Float32}` otherwise.
The matrix can be used as peek-ahead mask in transformers.

`dim=1` specifies the dimension in which the sequence length is
represented. For un-embedded data this is normally `1`, i.e. the
shape of x is [n_seq, n_mb]. After embedding the shape probably is
[depth, n_seq, n_mb].
"""
function mk_peek_ahead_mask(x; dim=1)

    n_seq = size(x)[dim]
    return convert2KnetArray(1 .- UpperTriangular(ones(n_seq, n_seq)))
end



"""
    function dot_prod_attn(q, k, v; mask=nothing)

Generic scaled dot product attention following the paper of
Vaswani et al., (2017), *Attention Is All You Need*.

### Arguments:
+ `q`: query of size `[depth, n_seq_q, ...]`
+ `k`: key of size `[depth, n_seq_v, ...]`
+ `v`: value of size `[depth, n_seq_v, ...]`
+ `mask`: mask for attention factors may have different shapes but must be
        broadcastable for addition to the scores tensor (which as the same size as
        alpha `[n_seq_v, n_seq_q, ...]`). In transformer context typical masks are one of:
        padding mask of size `[n_seq_v, ...]` or a peek-ahead mask of size `[n_seq_v, n_seq_v]`
        (which is only possible in case of self-attention when all seqencee lengths
        are identical).

`q, k, v` must have matching leading dimensions (i.e. same depth or embedding).
`k` and `v` must have the same sequence length.

### Return values:
+ `c`: context as alpha-weighted sum of values with size [depth, n_seq_v, ...]
+ `alpha`: attention factors of size [n_seq_v, n_seq_q, ...]
"""
function dot_prod_attn(q, k, v; mask=nothing)

    score = bmm(k, q, transA=true) ./ Float32(sqrt(size(k)[1]))  # [s_v x s_k x mb]

    if !isnothing(mask)
        score = score .+ mask * Float32(-1e9)
    end

    α = softmax(score, dims=1)
    c = bmm(v, α)
    return c, α
end
