using Pkg
Pkg.add("HDF5")
using HDF5

# test layers:
#
function test_layer_dense()
    i,o,mb = 50, 25, 16
    l = Dense(i,o, actf=sigm)
    x = convert2KnetArray(rand(Float32, i, mb))
    y = l(x)
    return size(y) == (o,mb)
end
function test_layer_pred()
    i,o,mb = 50, 25, 16
    l = Dense(i,o, actf=sigm)
    x = convert2KnetArray(rand(Float32, i, mb))
    y = l(x)
    return size(y) == (o,mb)
end

function test_dense_hdf5()
    h5file = h5open(joinpath("data", "dummykeras.h5"))
    l = Dense(h5file, "dense", trainable=true)

    i,o,mb = 3136, 128, 16
    x = convert2KnetArray(rand(Float32, i, mb))
    y = l(x)
    return size(y) == (o,mb)
end


function test_layer_linear()
    i,o,mb = 50, 25, 16
    l = Linear(i,o)
    x = convert2KnetArray(rand(Float32, i, 8, 8, mb))
    y = l(x)
    return size(y) == (o,8,8,mb)
end


function test_layer_conv()
    l = Conv(3,3,1,8)
    x = convert2KnetArray(rand(Float32, 28,28,1,16))
    y = l(x)
    return size(y) == (26,26,8,16)
end

function test_conv_hdf5()
    h5file = h5open(joinpath("data", "dummykeras.h5"))
    l = Conv(h5file, "conv2d", trainable=true) # 3x3x16, inp: 28x28x1
    x = convert2KnetArray(rand(Float32, 28,28,1,8))
    y = l(x)
    return size(y) == (28,28,16,8)
end


function test_layer_pool()
    l = Pool()
    x = convert2KnetArray(rand(Float32, 28,28,3,16))
    y = l(x)
    return size(y) == (14,14,3,16)
end

function test_layer_deconv()
    l = DeConv(3,3,8,8, padding=1, stride=2)
    x = rand(Float32, 28,28,8,16)
    x = convert2KnetArray(x)
    y = l(x)
    return size(y) == (55,55,8,16)
end

function test_layer_unpool()
    l = UnPool()
    x = rand(Float32, 14,14,3,16)
    x = convert2KnetArray(x)
    y = l(x)
    return size(y) == (28,28,3,16)
end

function test_layer_flat()
    l = Flat()
    x = rand(Float32, 10,10,3,16)
    x = convert2KnetArray(x)
    y = l(x)
    return size(y) == (300,16)
end

function test_layer_pyflat()
    l = PyFlat(python=true)
    x = rand(Float32, 10,10,3,16)
    x = convert2KnetArray(x)
    y = l(x)
    return size(y) == (300,16)
end



function test_layer_embed()
    l = Embed(16, 8)
    x = rand(1:16, 20)
    y = l(x)
    return size(y) == (8,20)
end



function test_layer_softmax()
    l = Softmax()
    x = rand(32)
    x = convert2KnetArray(x)
    y = l(x)
    return isapprox(sum(y), 1.0, atol=0.01)
end

function test_layer_dropout()
    l = Dropout(0.1)
    x = ones(100,100)
    x = convert2KnetArray(x)
    y = @diff l(x)
    return isapprox(sum(y .== 0), 1000, atol=100)
end

function test_layer_bn()
    l = BatchNorm()
    x = rand(16, 20)
    x = convert2KnetArray(x)
    y = @diff l(x)
    return isapprox(mean(y[1,:]), 0.0, atol=0.01)
end

function test_layer_ln()
    l = LayerNorm(16)
    x = rand(16, 20)
    x = convert2KnetArray(x)
    y = l(x)
    return isapprox(mean(y[:,1]), 0.0, atol=0.01)
end


function test_layer_seq_tagger()
    depth, seq, units, mb = 16, 5, 8, 10
    l = Recurrent(depth, units)
    x = convert2KnetArray(rand(depth, seq, mb))
    y = l(x, return_all=true)
    return size(y) == (units, seq, mb)
end

function test_layer_seq_classi()
    depth, seq, units, mb = 16, 5, 8, 10
    l = Recurrent(depth, units)
    x = convert2KnetArray(rand(depth, seq, mb))
    y = l(x)
    return size(y) == (units, mb)
end

function test_layer_rnn_loop()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    l = Recurrent(depth, units, allow_mask=true)
    x = convert2KnetArray(rand(depth, seq, mb))
    y = l(x, mask=mask)
    return size(y) == (units, mb)
end

mutable struct Peep <: RecurrentUnit
    w; w_r; b            # input
    w_i; w_ir; c_i; b_i  # input gate
    w_o; w_or; c_o; b_o  # output gate
    w_f; w_fr; c_f; b_f  # forget gate
    c                    # cell state
    h                    # last hidden state

    function Peep(i, n; o...)   # i: fan-in, n: num cells
        w = param(n, i);    w_r = param(n, n); b = param0(n)
        w_i = param(n, i); w_ir = param(n, n); c_i = param0(n); b_i = param0(n)
        w_o = param(n, i); w_or = param(n, n); c_o = param0(n); b_o = param0(n)
        w_f = param(n, i); w_fr = param(n, n); c_f = param0(n); b_f = param(n, init=ones)
        c = init0(n)
        h = init0(n)

        new(w, w_r, b, 
             w_i, w_ir, c_i, b_i, 
             w_o, w_or, c_o, b_o, 
             w_f, w_fr, c_f, b_f, 
             c, h)
    end
end
function (l::Peep)(x)
    
    # gates:
    #
    i_gate = sigm.(l.w_i * x .+ l.w_ir * l.h .+ l.c_i .* l.c .* l.b_i)
    o_gate = sigm.(l.w_o * x .+ l.w_or * l.h .+ l.c_o .* l.c .* l.b_o)
    f_gate = sigm.(l.w_f * x .+ l.w_fr * l.h .+ l.c_f .* l.c .* l.b_f)
    
    # cell state:
    #
    c_temp = tanh.(l.w * x .+ l.w_r * l.h .+ l.b)     
    l.c = c_temp .* i_gate .+ l.c .* f_gate
    
    # hidden state (output):
    #
    l.h = tanh.(l.c) .* o_gate
    return l.h
end

function test_layer_rnn_bi()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    layer = Recurrent(depth, units, u_type=:gru, bidirectional=true)  
    x = convert2KnetArray(rand(depth, seq, mb))
    y = layer(x)
    @show size(y)
    return size(y) == (2*units, mb)
end

function test_layer_rnn_bi_tagger()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    layer = Recurrent(depth, units, u_type=:gru, bidirectional=true)  
    x = convert2KnetArray(rand(depth, seq, mb))
    y = layer(x, return_all=true)
    @show size(y)
    return size(y) == (2*units, seq, mb)
end

function test_layer_rnn_bi_tagger_loop_Knet()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    layer = Recurrent(depth, units, u_type=:lstm, bidirectional=true)  
    x = convert2KnetArray(rand(depth, seq, mb))
    y = layer(x, return_all=true, mask=mask, h=0)
    @show size(y)
    return size(y) == (2*units, seq, mb)
end

function test_layer_Peep_rnn_bi()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    layer = Recurrent(depth, units, u_type=Peep, bidirectional=true)  
    x = convert2KnetArray(rand(depth, seq, mb))
    y = layer(x)
    @show size(y)
    return size(y) == (2*units, mb)
end



function test_layer_Peep_rnn()
    depth, seq, units, mb = 16, 5, 8, 10
    mask = convert2KnetArray(zeros(seq, mb))
    layer = Recurrent(depth, units, u_type=Peep)  
    x = convert2KnetArray(rand(depth, seq, mb))
    y = layer(x, mask=mask)
    #@show size(y)
    return size(y) == (units, mb)
end


function test_layer_H_rnn()
    depth, seq, units, mb = 16, 5, 8, 10
    l = Recurrent(depth, units)
    x = convert2KnetArray(rand(depth, seq, mb))
    y = l(x, h=0, c=0)

    h = get_hidden_states(l)
    c = get_cell_states(l)
    return size(h) == (units,mb,1) && size(c) == (units,mb,1)
end

function test_layer_RNN_2d()
    depth, seq, units, mb = 16, 5, 8, 1
    l = Recurrent(depth, units)
    x = convert2KnetArray(rand(depth, seq))
    y = l(x, h=0, c=0)

    h = get_hidden_states(l)
    c = get_cell_states(l)
    return size(h) == (units,mb,1)
end

function test_layer_K_rnn()
    depth, seq, units, mb = 16, 5, 8, 10
    l = RNN(depth, units; rnnType=:lstm, h=0, c=0)
    x = convert2KnetArray(rand(depth, mb, seq))
    y = l(x)

    h = get_hidden_states(l)
    c = get_cell_states(l)
    return size(h) == (units,mb,1) && size(c) == (units,mb,1)
end

function test_get_set_rnn()
    depth, seq, units, mb = 16, 5, 8, 10
    l = RNN(depth, units; rnnType=:lstm)

    set_hidden_states!(l, 0)
    set_cell_states!(l, 0)

    x = convert2KnetArray(rand(depth, mb, seq))
    y = l(x)

    reset_hidden_states!(l)
    reset_cell_states!(l)

    return get_hidden_states(l) == 0 && get_cell_states(l) == 0
end





function test_summary()

    ch = Chain(
            Dense(100,100),
            Linear(100,100),
            Conv(3, 3, 3, 100, padding=2),
            Pool(),
            DeConv(3,3, 100, 10, stride=2),
            UnPool(),
            Flat(),
            PyFlat(),
            PyFlat(python=false),
            Embed(100,10),
            Softmax(),
            Dropout(0.1),
            BatchNorm(trainable=true, channels=16),
            LayerNorm(128),
            Recurrent(100, 16),
            )
    n = print_network(ch)
    return n == 15
end

function test_print()

    ch = Chain(Conv(3, 3, 3, 100),
            Pool(),
            Conv(3,3,100,50))
    cl = Classifier(ch, Flat(), Linear(100,10))
    n = print_network(cl)

    return n == 5
end
