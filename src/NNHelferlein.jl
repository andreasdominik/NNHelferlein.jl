module NNHelferlein

import Base: iterate, length
# import Knet
import HDF5
import JLD2
using Statistics: mean
using ProgressMeter, Dates
using IterTools: ncycle, takenth
import CSV
# import OdsIO    # removed because of PyCall incompatibilities!
import DataFrames
import Random
using Printf
import CUDA
using Knet #: KnetArray, Param, @diff
import Images
import Augmentor
import MLDataUtils
using TensorBoardLogger, Logging

include("types.jl")
include("util.jl")
include("nets.jl")
include("layers.jl")
include("attn.jl")
include("funs.jl")
include("images.jl")
include("dataframes.jl")
include("train.jl")
include("imagenet.jl")


export NeuNet, Classifier, Regressor, Chain,          # chains
       ImageLoader, DataLoader, preproc_imagenet,
       get_class_labels,
       iterate, length,
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       DeConv, UnPool,
       Embed, Predictions,
       RSeqClassifer, RSeqTagger,
       Softmax, Dropout, BatchNorm,
       TensorDense, Linear,
       AttentionMechanism, AttnBahdanau,
       leaky_sigm,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches,
       mk_image_minibatch,
       tb_train!,
       predict_top5, predict_imagenet,
       predict,
       get_imagenet_classes,                    # images
       image2array, array2image, array2RGB,
       crop_array, init0,                       # utils
       blowup_array, recycle_array

end # module
