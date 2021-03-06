# using MLDatasets: MNIST
using Base.Iterators
import Pkg; Pkg.add("Augmentor"); using Augmentor
using DataFrames

function test_lenet()

    augm = CropSize(28,28)
    trn, vld = mk_image_minibatch("../data/flowers",
                4; split=true, fr=0.2,
                balanced=false, shuffle=true,
                train=true,
                aug_pipl=augm, pre_proc=preproc_imagenet)

    lenet = Classifier(Conv(5,5,3,20),
                    Pool(),
                    BatchNorm(),
                    Conv(5,5,20,50),
                    BatchNorm(trainable=true, channels = 50),
                    Pool(),
                    BatchNorm(trainable=true),
                    Flat(),
                    Dense(800,512),
                    Linear(512, 512, actf=relu),
                    Dense(512,3,actf=identity)
            )

    mdl = tb_train!(lenet, Adam, trn, vld, epochs=1,
            acc_fun=accuracy,
            eval_size=0.25, eval_freq=2, mb_loss_freq=100,
            tb_name="test_run", tb_text="NNHelferlein example")

    acc = accuracy(mdl, data=vld)

    tst = mk_image_minibatch("../data/flowers",
                4; split=false, fr=0.2,
                balanced=false, shuffle=true,
                train=false,
                aug_pipl=augm, pre_proc=nothing)
    p = predict_imagenet(mdl, tst, top_n=2)
    return acc isa Real && acc <= 1.0 &&
           isdir("logs") &&
           size(p) == (3,8)
end



function acc_fun(mdl; data=data)
    a = 0.0
    for (x,y) in data
        a += mean(abs2, mdl(x) .- y)
    end
    return a/length(data)
end

function test_mlp()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=collect(range(0, 1, length=16)))

        mb = dataframe_minibatches(trn, size=4)

        mlp = Regressor(Dense(8,8, actf=relu),
                         Dense(8,8),
                         Dense(8,1, actf=identity))

        mlp = tb_train!(mlp, Adam, mb, epochs=10, acc_fun=nothing,
                lr=0.001, lr_decay=0.0001, lrd_freq=5)


        mlp = tb_train!(mlp, Adam, mb, epochs=10, acc_fun=acc_fun,
                lr=0.001, lr_decay=0.0001, lrd_freq=5)

        acc = NNHelferlein.calc_acc(mlp, acc_fun, data=mb)
        return acc isa Real
end

function test_signatures()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=collect(range(0, 1, length=16)))

        mb = dataframe_minibatches(trn, size=4)

        mlp = Regressor(Dense(8,8, actf=relu),
                         Dense(8,8),
                         Dense(8,1, actf=identity))
        mlp = tb_train!(mlp, Adam, mb, epochs=1, acc_fun=nothing)

        y = mlp(rand(Float32, 8,4))
        test_sign = size(y) == (1,4)

        loss = mlp(rand(Float32, 8,4), rand(Float32, 1,4))
        test_sign = test_sign && typeof(loss) <: Real

        loss = mlp(mb)
        test_sign = test_sign && typeof(loss) <: Real

        loss = mlp(first(mb))
        test_sign = test_sign && typeof(loss) <: Real

        x,y = first(mb)
        loss = mlp(x,y)
        test_sign = test_sign && typeof(loss) <: Real

        return test_sign
end


function test_decay_cp()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=collect(range(0, 1, length=16)))

        mb = dataframe_minibatches(trn, size=4)

        chain = Chain(Dense(8,8, actf=relu),
                      Dense(8,8))
        mlp = Regressor(chain,
                        Dense(8,1, actf=identity))

        mlp = tb_train!(mlp, Adam, mb, epochs=2, acc_fun=nothing,
                cp_freq=1, lr=0.01, lr_decay=0.99, l2=1e-6)
        acc = NNHelferlein.calc_acc(mlp, acc_fun, data=mb)
        return acc isa Real
end


function test_hamming()

        p = [0  0  0  0  0
             4  4  3  4  5
             2  2  2  2  5
             1  3  1  3  3
             0  0  0  0  0]
        t = [0  0  0  0  0
             4  4  5  3  4
             3  1  2  4  4
             2  5  5  4  2
             0  0  0  0  0]

        acc = hamming_dist(p,t)
        hd = hamming_dist(p,t, accuracy=true)

        return acc ≈ 2.4 && hd ≈ 0.2
end
