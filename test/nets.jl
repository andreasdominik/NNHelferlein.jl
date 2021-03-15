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
                    Conv(5,5,20,50),
                    BatchNorm(;trainable=true, channels = 50),
                    Pool(),
                    Flat(),
                    Dense(800,512),
                    Linear(512, 512, actf=relu),
                    Predictions(512,3)
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




function test_mlp()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=collect(range(0, 1, length=16)))

        mb = dataframe_minibatches(trn, size=4)

        mlp = Regressor(Dense(8,8, actf=relu),
                         Dense(8,8),
                         Predictions(8,1))

        mlp = tb_train!(mlp, Adam, mb, epochs=1, acc_fun=nothing)
        acc = NNHelferlein.calc_acc(mlp, (x,y)->mean(abs2, x-y), data=mb)
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
                         Predictions(8,1))
        mlp = tb_train!(mlp, Adam, mb, epochs=1, acc_fun=nothing)

        y = mlp(rand(Float32, 8,4))
        test_sign = size(y) == (1,4)

        loss = mlp(rand(Float32, 8,4), rand(Float32, 1,4))
        test_sign = test_sign && typeof(loss) <: Real

        loss = mlp(mb)
        test_sign = test_sign && typeof(loss) <: Real

        loss = mlp(first(mb))
        test_sign = test_sign && typeof(loss) <: Real

        return test_sign
end
