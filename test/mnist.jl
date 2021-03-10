using MLDatasets: MNIST
using Base.Iterators

function test_lenet()

    xtrn,ytrn = MNIST.traindata(Float32)
    ytrn[ytrn.==0] .= 10
    dtrn = take(minibatch(xtrn, ytrn, 100; xsize = (28,28,1,:)), 10)

    xvld,yvld = MNIST.testdata(Float32)
    yvld[yvld.==0] .= 10
    dvld = take(minibatch(xvld, yvld, 100; xsize = (28,28,1,:)), 10)

    lenet = Classifier(Conv(5,5,1,20),
                    Pool(),
                    Conv(5,5,20,50),
                    Pool(),
                    Flat(),
                    Dense(800,512),
                    Linear(512, 512, actf=relu),
                    Predictions(512,10)
            )

    mdl = tb_train!(lenet, Adam, dtrn, dvld, epochs=1,
            acc_fun=accuracy,
            eval_size=0.25, eval_freq=2, mb_loss_freq=100,
            tb_name="example_run", tb_text="NNHelferlein example")

    return mdl isa Classifier
end
