module NNHelferlein

import Base.iterate, Base.length
import Knet
using Knet: KnetArray, Param
import CUDA
import CSV
import OdsIO
import DataFrames
import Random
import Images
import Augmentor
import MLDataUtils


include("nets")
include("layers")

export NeuNet, Classifier, Regressor,           # chains
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       Embed,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches


end # module
