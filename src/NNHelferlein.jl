module NNHelferlein

import Base: iterate, length, summary
using LinearAlgebra
using Unicode
import ZipFile
import HDF5
import JLD2
using Statistics: mean, std
using ProgressMeter, Dates
using IterTools: ncycle, takenth
import CSV
# import OdsIO    # removed because of PyCall incompatibilities!
import DataFrames
import Random
using Printf
import CUDA
using Knet #: KnetArray, Param, @diff
import Images, Colors
import Augmentor
import MLDataUtils
using TensorBoardLogger, Logging

include("types.jl")
include("util.jl")
include("nets.jl")
include("layers.jl")
include("attn.jl")
include("funs.jl")
include("transformers.jl")
include("images.jl")
include("dataframes.jl")
include("texts.jl")
include("train.jl")
include("imagenet.jl")


DATA_DIR = normpath(joinpath(dirname(pathof(@__MODULE__)), "..", "data"))

export DNN, Classifier, Regressor, Chain,          # chains
       ImageLoader, DataLoader, preproc_imagenet,
       get_class_labels,
       iterate, length,
       Layer, Dense, Conv, Pool, Flat, PyFlat,         # layers
       DeConv, UnPool,
       Embed,
       RSeqClassifier, RSeqTagger,
       hidden_states, cell_states,
       Softmax, Dropout, BatchNorm, LayerNorm,
       Linear,
       AttentionMechanism, AttnBahdanau,
       AttnLuong, AttnDot, AttnLocation,
       AttnInFeed,
       leaky_sigm, leaky_relu, leaky_tanh,
       PositionalEncoding, positional_encoding_sincos,  # transformers
       mk_padding_mask, mk_peek_ahead_mask,
       dot_prod_attn,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches, mk_class_ids,
       mk_image_minibatch,
       tb_train!,
       predict_top5, predict_imagenet,
       predict, hamming_dist,
       get_imagenet_classes,                    # images
       image2array, array2image, array2RGB,
       WordTokenizer,                           # texts
       get_tatoeba_corpus,
       seq_minibatch, seq2seq_minibatch,
       crop_array, init0, convert2KnetArray,             # utils
       blowup_array, recycle_array,
       de_embed,
       print_network,
       DATA_DIR

end # module
