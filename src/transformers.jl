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
    function mk_padding_mask(x; pad=TOKEN_PAD, add_dims=false)

Make a padding mask; i.e. return an Array of type
`KnetArray{Float32}` (or `Array{Float32}`) similar to `x` but with
two additional dimension of size 1 in the middle (this will represent the
2nd seq_len and the number of heads) in multi-head attention
and the
value `1.0` at each position where `x` is `pad` and `0.0` otherwise.

The function can be used for creating padding masks for attention
mechanisms.

### Arguments:
+ `x`: Array of sequences (typically a matrix with n_cols sequences
    of length n_rows)
+ `pad`: value for the token to be masked
+ `add_dims`: if `true`, 2 additional dimensions are inserted to 
    return a 4-D-array as needed for transformer architectures. Otherwise
    the size of teh returned array is similar to x.
"""
function mk_padding_mask(x; pad=TOKEN_PAD, add_dims=false)

    if add_dims
       return reshape(convert2KnetArray(x .== pad), size(x)[1],1,1,size(x)[2])
    else
       return convert2KnetArray(x .== pad)
    end
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


"""
    struct MultiHeadAttn <: Layer

Multi-headed attention layer, designed following the Vaswani, 2017 paper.

### Constructor:

    MultiHeadAttn(depth, n_heads) 

+ `depth`: Embedding depth
+ `n_heads`: number of heads for the attention.

### Signature:

    function(mha::MultiHeadAttn)(q, k, v; mask=nothing)

`q, k, v` are 3-dimensional tensors of the same size
([depth, seq_len, n_minibatch]) and the optional mask must be of 
size [seq_len, n_minibatch] and mark masked positions with 1.0.

"""
mutable struct MultiHeadAttn
    dense_q        # x -> q
    dense_k        # x -> K
    dense_v        # x -> v
    depth          # embedding
    n_heads        #
    h_depth        # embedding / heads
    dense_out      # out layer
    MultiHeadAttn(depth, n_heads) = new(Linear(depth, depth), Linear(depth, depth), 
                                        Linear(depth, depth),
                                        depth, n_heads, depth÷n_heads,
                                        Linear(depth, depth))
end

function(mha::MultiHeadAttn)(q, k, v; mask=nothing)

    q = mha.dense_q(q)      # [depth, n_seq, n_mb]
    k = mha.dense_k(k)
    v = mha.dense_v(v)

    q = separate_heads(q, mha.n_heads)      # [depth/n, n_seq, n_heads, n_mb]
    k = separate_heads(k, mha.n_heads)
    v = separate_heads(v, mha.n_heads)

    c, α = dot_prod_attn(q, k, v, mask=mask)  # c: [depth/n, n_seq, n_heads, n_mb]
                                              # α: [n_seq, n_seq, n_heads, n_mb]
    c = merge_heads(c)                        # [depth, n_seq_ n_mb]
    return mha.dense_out(c), α
end




"""
    function separate_heads(x, n)

Helper function for multi-headed attention mechanisms: 
an additional second dimension is added to a tensor of minibatches
by splitting the first (i.e. depth).
"""
function separate_heads(x, n)
    depth, seq, mb = size(x)
    mh_depth = depth ÷ n
    x = reshape(x, mh_depth, n, :, mb)     # split depth in 2 separate dims for heads
    return permutedims(x, (1,3,2,4))       # bring seq-len back to 2nd dim
end

"""
    function merge_heads(x)

Helper to merge the result of multi-headed attention back to full
depth .
"""
function merge_heads(x)
    mh_depth, seq, n, mb = size(x)
    depth = mh_depth * n
    x = permutedims(x, (1,3,2,4))          # bring heads back to 2nd dim
    return reshape(x, depth, :, mb)        # merde depth and heads (dims 1 and 2) into 1st dim
end




