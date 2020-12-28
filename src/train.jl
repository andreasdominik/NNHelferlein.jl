"""
    function tb_train!(mdl, opti, trn; epoch=1, vld=nothing, eval_size=0.1,
                      mb_loss_freq=100, eval_freq=1,
                      tb_dir="./tensorboard_logs", tb_name="run")

Train function with TensorBoard integration. TB logs are written with
the TensorBoardLogger.jl package.
The model is updated (in-place) and the trained model is returned.

### Arguments:
+ `mdl`: model; i.e. forward-function for the net
+ `opti`: Knet-stype optimiser iterator
+ `trn`: training data; iterator to provide (x,y)-tuples with
        minibatches

### Keyword arguments:
+ `epoch`: number of epochs to train
+ `vld`: validation data
+ `eval_size`: fraction of validation data to be used for calculating
        loss and accuracy for train and validation data during training.
+ `eval_freq`: frequency of evaluation; default=1 means evaluation is
        calculated after each epoch. With eval_freq=10 eveluation is
        calculated 10 times per epoch.
+ `mb_loss_freq`: frequency of training loss reporting. default=100
        means that 100 loss-values per epoch will be logged to TensorBoard.
        If mb_loss_freq is greater then the number of minibatches,
        loss is logged for each minibatch.

### Definition of TensorBoard log-directory:
TensorBoard log-directory is created from 3 parts:
`tb_dir/tb/name/<current date time>`.

+ `tb_dir`: root directory for tensorborad logs.
+ `tb_name`: name of training run.
"""
function tb_train!(mdl, opti, trn; epoch=1, vld=nothing, eval_size=0.1,
                  mb_loss_freq=100, eval_freq=1,
                  tb_dir="./tensorboard_logs", tb_name="run")

    # use every n-th mb for evaluation (based on vld if defined):
    #
    n_trn = length(trn)
    n_vld = vld != nothing ? length(vld) : 1

    if vld == nothing
        n_eval = Int(ceil(n_trn * eval_size))
    else
        n_eval = Int(ceil(n_vld * eval_size))
    end
    nth_trn = Int(cld(n_trn, n_eval))
    nth_vld = Int(cld(n_vld, n_eval))

    eval_freq = Int(cld(n_trn, eval_freq))
    mb_loss_freq = Int(cld(n_trn, mb_loss_freq))

    println("Training $epoch epochs with $n_trn minibatches/epoch (and $n_vld validation mbs).")
    println("Evaluation is performed every $eval_freq minibatches (with $n_eval mbs).")

    # Tensorboard logger:
    #
    tb_log_dir = joinpath(tb_dir, tb_name, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    tbl = TensorBoardLogger.TBLogger(tb_log_dir, min_level=Logging.Info)

    #     layout_loss = Dict("Losses" => Dict("Train and valid loss" => (tb_multiline, ["train loss", "valid loss"])))
    #     TensorBoardLogger.log_custom_scalar(tbl, layout_loss)
    #     layout_acc = Dict("Accuracies" => Dict("Train and valid accuracy" => (tb_multiline, ["train acc", "valid acc"])))
    #     TensorBoardLogger.log_custom_scalar(tbl, layout_acc)
    #     calc_and_report_loss_acc(mdl, takenth(trn, nth_trn), takenth(vld, nth_vld), tbl, 0)

    # Training:
    #
    mb_losses = Float32[]
    @showprogress for (i, mb_loss) in enumerate(adam(lenet, ncycle(dtrn,1)))

        push!(mb_losses, mb_loss)
        if (i % eval_freq) == 0
            calc_and_report_loss_acc(mdl, takenth(trn, nth_trn), takenth(vld, nth_vld), tbl, eval_freq)
        end

        if (i % mb_loss_freq) == 0
        #     println("             write loss at $i: $(mean(mb_losses))")
            TensorBoardLogger.log_value(tbl, "Minibatch loss (epoch = $n_trn steps)", mean(mb_losses), step=i)
            mb_losses = Float32[]
        end
    end
    return mdl
end

# Helper to calc loss and acc with only ONE forward run:
#
function loss_and_acc(mdl, data)

    acc = nll = len = 0.0
    for (x,y) in data
        preds = mdl(x)
        len += length(y)

        acc += Knet.accuracy(preds,y, average=false)[1]
        nll += Knet.nll(preds,y, average=false)[1]
    end

    return nll/len, acc/len
end


# Helper for TensorBoardLogger:
#
function calc_and_report_loss_acc(mdl, trn, vld, tbl, step)
        loss_trn, acc_trn = loss_and_acc(mdl, trn)
        loss_vld, acc_vld = loss_and_acc(mdl, vld)
        #     println("eval at $i: loss = $loss_trn, $loss_vld; acc =  = $acc_trn, $acc_vld")

        with_logger(tbl) do
            @info "Evaluation Loss (every $step steps)" train=loss_trn valid=loss_vld log_step_increment=step
            @info "Evaluation Accuracy (every $step steps)" train=acc_trn valid=acc_vld log_step_increment=0
    end
end
