using Knet, NNHelferlein
using Test

# test Conv, Dense and tb_train():
#
include("mnist.jl")
@test test_lenet()
