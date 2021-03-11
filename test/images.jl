# using MLDatasets: MNIST
using Base.Iterators
import Pkg; Pkg.add("Augmentor"); using Augmentor

function test_image_loader()
    augm = CropSize(28,28)
    trn, vld = mk_image_minibatch("data/flowers",
                4; split=true, fr=0.2,
                balanced=false, shuffle=true,
                train=true,
                aug_pipl=augm, pre_proc=nothing)

    return size(first(trn)[1]) == (28,28,3,4)
end

# function test_lenet()
#
#     MNIST.download(;i_accept_the_terms_of_use=true)
#     xtrn,ytrn = MNIST.traindata(Float32)
#     ytrn[ytrn.==0] .= 10
#     dtrn = take(minibatch(xtrn, ytrn, 100; xsize = (28,28,1,:)), 10)
#
#     xvld,yvld = MNIST.testdata(Float32)
#     yvld[yvld.==0] .= 10
#     dvld = take(minibatch(xvld, yvld, 100; xsize = (28,28,1,:)), 10)
#
#     lenet = Classifier(Conv(5,5,1,20),
#                     Pool(),
#                     Conv(5,5,20,50),
#                     Pool(),
#                     Flat(),
#                     Dense(800,512),
#                     Linear(512, 512, actf=relu),
#                     Predictions(512,10)
#             )
#
#     mdl = tb_train!(lenet, Adam, dtrn, dvld, epochs=1,
#             acc_fun=accuracy,
#             eval_size=0.25, eval_freq=2, mb_loss_freq=100,
#             tb_name="example_run", tb_text="NNHelferlein example")
#
#     return mdl isa Classifier
# end
#
#
# function test_mlp()
#
#     xtrn,ytrn = MNIST.traindata(Float32)
#     ytrn[ytrn.==0] .= 10
#     dtrn = take(minibatch(xtrn, ytrn, 100; xsize = (28,28,1,:)), 10)
#
#     xvld,yvld = MNIST.testdata(Float32)
#     yvld[yvld.==0] .= 10
#     dvld = take(minibatch(xvld, yvld, 100; xsize = (28,28,1,:)), 10)
#
#     lenet = Classifier(Conv(5,5,1,20),
#                     BatchNorm(trainable=true, channels=28*28)
#                     PyFlat(python=true),
#                     Dense(800,512),
#                     Linear(512, 512, actf=relu),
#                     Predictions(512,10)
#             )
#
#     mdl = tb_train!(lenet, Adam, dtrn, dvld, epochs=1,
#             acc_fun=accuracy,
#             eval_size=0.25, eval_freq=2, mb_loss_freq=100,
#             tb_name="example_run", tb_text="NNHelferlein example")
#
#     return mdl isa Classifier
# end
