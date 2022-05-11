module NNHelferlein
# import Pkg; Pkg.add("Images")
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
import MLBase: confusmat
# TODO: tidy-up!

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
include("acc.jl")
include("imagenet.jl")
include("data.jl")


DATA_DIR = normpath(joinpath(dirname(pathof(@__MODULE__)), "..", "data"))

export DNN, Classifier, Regressor, Chain, VAE,          # chains
       DataLoader, SequenceData, PartialIterator,
       RecurrentUnit,
       add_layer!, 
       split_minibatches,
       ImageLoader, preproc_imagenet,
       get_class_labels,
       iterate, length,
       Layer, Dense, Conv, Pool, Flat, PyFlat,         # layers
       DeConv, UnPool,
       Embed,
       Recurrent,
       get_hidden_states, get_cell_states,
       set_hidden_states!, set_cell_states!,
       reset_hidden_states!, reset_cell_states!,
       Softmax, Dropout, BatchNorm, LayerNorm,
       Linear,
       AttentionMechanism, AttnBahdanau,
       AttnLuong, AttnDot, AttnLocation,
       AttnInFeed,
       leaky_sigm, leaky_relu, leaky_tanh,
       PositionalEncoding, positional_encoding_sincos,  # transformers
       mk_padding_mask, mk_peek_ahead_mask,
       dot_prod_attn, MultiHeadAttn,
       separate_heads, merge_heads,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches, mk_class_ids,
       MBNoiser,
       mk_image_minibatch,
       tb_train!,
       predict_top5, predict_imagenet,
       predict, hamming_dist, hamming_acc,
       peak_finder_acc,
       get_imagenet_classes,                    # images
       image2array, array2image, array2RGB,
       clean_sentence, WordTokenizer,                    # texts
       get_tatoeba_corpus,
       sequence_minibatch, pad_sequence, truncate_sequence, 
       TOKEN_START, TOKEN_END, TOKEN_PAD, TOKEN_UNKOWN,
       crop_array, init0, 
       convert2CuArray, emptyCuArray, ifgpu,
       convert2KnetArray, emptyKnetArray,
       blowup_array, recycle_array,
       de_embed,
       print_network,
       DATA_DIR, #, download_example_data, download_pretrained
       confusion_matrix,
       dataset_mit_nsr

end # module
