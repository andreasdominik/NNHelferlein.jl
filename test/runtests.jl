using Pkg
Pkg.add(url="https://github.com/denizyuret/AutoGrad.jl.git")

using Knet, NNHelferlein, CUDA
using Images, Colors
import Pkg; Pkg.add("Test"); using Test
using Statistics: mean



# test attention mechanisms:
#
@show include("attn.jl")
@test test_attn(AttnBahdanau)
@test test_attn(AttnLuong)
@test test_attnDot()
@test test_attnLocation()
@test test_attnInFeed()

@test test_dpa()
@test test_masks()
@test test_dotp_attn()



# data loader:
#
@show include("data.jl")
@test test_read_df()
@test test_df_loader()
@test test_df_split()
@test test_df_class_ids()
@test test_df_minibatch()
@test test_df_errors()
@test test_noiser()
@test test_split_mbs()
@test test_partial_itr()

# image loader:
#
@show include("images.jl")
#@test test_image_loader()   # problem with Images.load()?
#@test test_image_preload()
@test test_image2arr()
@test test_array2image()
@test test_preproc_imagenet()
@test test_in_classes()

# text loader:
#
@test test_tokenizer()
@test test_seq_mb()
@test test_seq_mb_xy()
@test test_tatoeba()
@test test_seq2seq_mb()

@test test_pad()
@test test_trunc()


# test all layers:
#
@show include("layers.jl")
@test test_layer_dense()
@test test_layer_pred()
@test test_dense_hdf5()
@test test_layer_linear()

@test test_layer_conv()
@test test_conv_hdf5()
@test test_layer_pool()
@test test_layer_deconv()
@test test_layer_unpool()
@test test_layer_flat()
@test test_layer_pyflat()
@test test_layer_embed()
@test test_layer_softmax()
@test test_layer_dropout()
@test test_layer_bn()
@test test_layer_ln()

@test test_layer_seq_tagger()
@test test_layer_seq_classi()
@test test_layer_rnn_loop()

@test test_layer_rnn_bi()
@test test_layer_rnn_bi_tagger()
@test test_layer_rnn_bi_tagger_loop_Knet()
@test test_layer_Peep_rnn_bi()
@test test_layer_Peep_rnn()

@test test_layer_H_rnn()
@test test_layer_K_rnn()
@test test_layer_RNN_2d()
@test test_get_set_rnn()


# acc_funs:
#
@show include("acc.jl")
@test test_peak_finder()
@test test_peak_finder_acc()
@test test_hamming()
@test test_hamming_acc()


# utils:
#
@show include("util.jl")
@test test_crop_array()
@test test_init0()
@test test_convertKA()
@test test_blowup()
@test test_recycle()
@test test_de_embed()

# other funs:
#
@show include("funs.jl")
@test test_leaky()


# test Conv, Dense and tb_train():
#
@show include("nets.jl")
@test test_symbolic_api()
@test test_lenet()
@test test_mlp()
@test test_signatures()
@test test_decay_cp()

@test test_vae()


@test test_summary()
@test test_print()
