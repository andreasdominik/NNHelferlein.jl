# test attn:
#
using Knet, NNHelferlein

depth, mb, t = 3, 4, 6

h_enc = convert2KnetArray(randn(Float32, depth, mb, t))
h_t = convert2KnetArray(randn(Float32, depth, mb))

function test_attn_reset()
    a = AttnBahdanau(depth, depth)

    a.projections = rand(depth, depth)
    a()   # reset = true

    return !isnothing(a.projections)
end

function test_attn(attn)
    a = attn(depth, depth)
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnDot()
    a = AttnDot()
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnLocation()
    a = AttnLocation(t, depth)
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnInFeed()
    a = AttnInFeed(t, depth, depth)
    c,α = a(h_t, init0(depth), h_enc)

    return size(c) == (3,4)
end


import NNHelferlein: resize_attn_mask
function test_attn_resize()
    size_3 = size(resize_attn_mask(rand(5,5,5))) # 5,5,5
    size_2 = size(resize_attn_mask(rand(5,5))) # 1,5,5
    size_1 = size(resize_attn_mask(rand(5))) # 1,1,5
    size_o = size(resize_attn_mask(rand(5,5,5,5))) # 1,

    return size_3 == (5,5,5) &&
           size_2 == (1,5,5) &&
           size_1 == (1,1,5) &&
           size_o == (1,)
end
    


# transformer tests:
#
function test_dpa()
    kqv = rand(Float32, 3,8,10)
    c,a = dot_prod_attn(kqv, kqv, kqv)

    return size(c) == (3,8,10) && size(a) == (8,8,10)
end

function test_masks()
    seqs = rand(1:16, 4,6)
    el = Embed(16,8)
    seqs_e = el(seqs)

    pl = PositionalEncoding()
    pos_enc = pl(seqs)   # assert 4x6

    peek_ah = mk_peek_ahead_mask(seqs)
    padd = mk_padding_mask(seqs, add_dims=true)

    return size(pos_enc) == (4,6) &&
           size(peek_ah) == (4,4) &&
           size(padd) == (4,1,1,6)
end


function test_dotp_attn()

    function separate_heads(x, n)
        depth, seq, mb = size(x)
        mh_depth = depth ÷ n
        x = reshape(x, mh_depth, n, :, mb)     # split depth in 2 separate dims for heads
        return permutedims(x, (1,3,2,4))       # bring seq-len back to 2nd dim
    end

    seqs = rand(1:16, 4,6)
    el = Embed(16,8)
    emb = el(seqs)
    heads = separate_heads(emb, 2)
    padd = mk_padding_mask(seqs, add_dims=true)

    dpa = dot_prod_attn(heads, heads, heads, mask=padd)

    return size(dpa[1]) == (4,4,2,6)
end


function test_mha()
    mha = MultiHeadAttn(512, 8)
    x = convert2CuArray(randn(Float32, 512, 16, 64)) 
    c,a = mha(x,x,x)

    return size(c) == (512, 16, 64) && size(a) == (16, 16, 8, 64)
end


