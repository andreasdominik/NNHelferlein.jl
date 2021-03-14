using HDF5

# test layers:
#
function test_layer_dense()
    i,o,mb = 50, 25, 16
    l = Dense(i,o, actf=sigm)
    x = rand(Float32, i, mb)
    y = l(x)
    return size(y) == (o,mb)
end

function test_dense_hdf5()
    h5file = h5open("../data/testdata/dummykeras.h5")
    l = Dense(h5file, "dense", trainable=true)

    i,o,mb = 3136, 128, 16
    x = rand(Float32, i, mb)
    y = l(x)
    return size(y) == (o,mb)
end


function test_layer_linear()
    i,o,mb = 50, 25, 16
    l = Linear(i,o)
    x = rand(Float32, i, 8, 8, mb)
    y = l(x)
    return size(y) == (o,8,8,mb)
end


function test_layer_conv()
    l = Conv(3,3,1,8)
    x = rand(Float32, 28,28,1,16)
    y = l(x)
    return size(y) == (26,26,8,16)
end

function test_conv_hdf5()
    h5file = h5open("../data/testdata/dummykeras.h5")
    l = Conv(h5file, "conv2d", trainable=true) # 3x3x16, inp: 28x28x1
    x = rand(Float32, 28,28,1,8)
    y = l(x)
    return size(y) == (28,28,16,8)
end


function test_layer_pool()
    l = Pool()
    x = rand(Float32, 28,28,3,16)
    y = l(x)
    return size(y) == (14,14,3,16)
end

function test_layer_deconv()
    l = DeConv(3,3,8,8, padding=1, stride=2)
    x = rand(Float32, 28,28,8,16)
    y = l(x)
    return size(y) == (55,55,8,16)
end

function test_layer_unpool()
    l = UnPool()
    x = rand(Float32, 14,14,3,16)
    y = l(x)
    return size(y) == (28,28,3,16)
end

function test_layer_flat()
    l = Flat()
    x = rand(Float32, 10,10,3,16)
    y = l(x)
    return size(y) == (300,16)
end

function test_layer_pyflat()
    l = PyFlat(python=true)
    x = rand(Float32, 10,10,3,16)
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
    y = l(x)
    return isapprox(sum(y), 1.0, atol=0.01)
end

function test_layer_dropout()
    l = Dropout(0.1)
    x = ones(100,100)
    y = @diff l(x)
    return isapprox(sum(y .== 0), 1000, atol=100)
end

function test_layer_bn()
    l = BatchNorm()
    x = rand(16, 20)
    y = @diff l(x)
    return isapprox(mean(y[1,:]), 0.0, atol=0.01)
end

function test_layer_ln()
    l = LayerNorm(16)
    x = rand(16, 20)
    y = l(x)
    return isapprox(mean(y[:,1]), 0.0, atol=0.01)
end
