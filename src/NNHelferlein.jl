module NNHelferlein

import Base: iterate, length
# import Knet
using Knet #: KnetArray, Param, @diff
import CUDA
import CSV
import OdsIO
import DataFrames
import Random
import Images
import Augmentor
import MLDataUtils
using Statistics: mean
using ProgressMeter, Dates
using TensorBoardLogger, Logging
using IterTools: ncycle, takenth
using HDF5
import JLD2
using Printf

include("types.jl")
include("util.jl")
include("nets.jl")
include("layers.jl")
include("funs.jl")
include("images.jl")
include("dataframes.jl")
include("train.jl")
include("imagenet.jl")


export NeuNet, Classifier, Regressor,           # chains
       ImageLoader, DataLoader, preproc_imagenet,
       get_class_labels,
       iterate, length,
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       Embed, Predictions,
       Dropout, BatchNorm,
       leaky_sigm,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches,
       mk_image_minibatch,
       tb_train!, predict_top5, predict_imagenet,
       predict,
       get_imagenet_classes

end # module
