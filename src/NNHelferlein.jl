module NNHelferlein

using Knet
using CUDA

include("nets")
include("layers")

export NeuNet, Classifier, Regressor,           # chains
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       Embed


end # module
