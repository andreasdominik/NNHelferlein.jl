using Knet, NNHelferlein
using Test

# test Conv, Dense and tb_train():
#
include("mnist.jl")
@test test_lenet()


# test attention mechanisms:
#
include("attn.jl")
@test test_attn(AttnBahdanau)
@test test_attn(AttnLuong)
@test test_attnDot()
@test test_attnLocation()
@test test_attnInFeed()


# texts:
#
@test test_tokenizer()
