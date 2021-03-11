using Knet, NNHelferlein
import Pkg; Pkg.add("Test"); using Test

# test Conv, Dense and tb_train():
#
# include("mnist.jl")
# @test test_lenet()


# test attention mechanisms:
#
include("attn.jl")
@test test_attn(AttnBahdanau)
@test test_attn(AttnLuong)
@test test_attnDot()
@test test_attnLocation()
@test test_attnInFeed()

@test test_dpa()


# texts:
#
include("texts.jl")
@test test_tokenizer()
@test test_seq_mb()
