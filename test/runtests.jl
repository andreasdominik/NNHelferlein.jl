using Knet, NNHelferlein
import Pkg; Pkg.add("Test"); using Test

# test Conv, Dense and tb_train():
#
include("nets.jl")
@test test_image_loader()
@test test_lenet()
@test test_df_loader()
@test test_mlp()


# test attention mechanisms:
#
include("attn.jl")
@test test_attn(AttnBahdanau)
@test test_attn(AttnLuong)
@test test_attnDot()
@test test_attnLocation()
@test test_attnInFeed()

@test test_dpa()



# data loader:
#
include("data.jl")
@test test_read_df()
@test test_df_split()
@test test_df_class_ids()
@test test_df_minibatch()

# image loader:
# tested in nets.jl!
#
# imagenet:
#
test_preproc_imagenet()
test_in_classes()

# text loader:
#
@test test_tokenizer()
@test test_seq_mb()
@test test_tatoeba()
@test test_seq2seq_mb()


# test all layers:
#
@test test_layer_dense()
@test test_dense_hdf5()
@test test_layer_linear()

@test test_layer_conv()
@test test_conv_hdf5()
@test test_layer_pool()
