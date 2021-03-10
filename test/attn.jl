# test attn:
#
using Knet, NNHelferlein

depth, mb, t = 3, 4, 6

h_enc = randn(Float32, depth, mb, t)
h_t = randn(Float32, depth, mb)

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



# transformer tests:
#
