module NNHelferlein

using Knet: param, param0, sigm, mat,
            conv4, pool
import CUDA
import CSV
import OdsIO

include("nets")
include("layers")

export NeuNet, Classifier, Regressor,           # chains
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       Embed


end # module
