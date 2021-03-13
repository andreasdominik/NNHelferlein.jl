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
