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
