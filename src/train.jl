"""
    function tb_train!(mdl, opti, trn; epochs=1, vld=nothing, eval_size=0.2,
                      mb_loss_freq=100, eval_freq=1,
                      cp_freq=1, cp_dir="checkpoints",
                      tb_dir="tensorboard_logs", tb_name="run",
                      tb_text=\"\"\"Description of tb_train!() run.\"\"\")

Train function with TensorBoard integration. TB logs are written with
the TensorBoardLogger.jl package.
The model is updated (in-place) and the trained model is returned.

### Arguments:
+ `mdl`: model; i.e. forward-function for the net
+ `opti`: Knet-stype optimiser iterator
+ `trn`: training data; iterator to provide (x,y)-tuples with
        minibatches

### Keyword arguments:
+ `epochs=1`: number of epochs to train
+ `vld=nothing`: validation data
+ `eval_size=0.2`: fraction of validation data to be used for calculating
        loss and accuracy for train and validation data during training.
+ `eval_freq=1`: frequency of evaluation; default=1 means evaluation is
        calculated after each epoch. With eval_freq=10 eveluation is
        calculated 10 times per epoch.
+ `mb_loss_freq=100`: frequency of training loss reporting. default=100
        means that 100 loss-values per epoch will be logged to TensorBoard.
        If mb_loss_freq is greater then the number of minibatches,
        loss is logged for each minibatch.
+ `cp_freq=1`: frequency of model checkpoints written to disk.
        Default is to write the model after each epoch with
        name `model`.
+ `cp_dir="checkpoints"`: directory for checkpoints.

### TensorBoard kw-args:
TensorBoard log-directory is created from 3 parts:
`tb_dir/tb/name/<current date time>`.

+ `tb_dir="tensorboard_logs"`: root directory for tensorborad logs.
+ `tb_name="run"`: name of training run. `tb_name` will be used as
        directory name and should not include whitespace
+ `tb_text`:  description
        to be included in the TensorBoard log.
"""
function tb_train!(mdl, opti, trn; epochs=1, vld=nothing, eval_size=0.1,
                  mb_loss_freq=100, eval_freq=1,
                  cp_freq=1, cp_dir="checkpoints",
                  tb_dir="logs", tb_name="run",
                  tb_text="""Description of tb_train!() run.""")

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

    eval_nth = Int(cld(n_trn, eval_freq))
    mb_loss_nth = Int(cld(n_trn, mb_loss_freq))

    println("Training $epochs epochs with $n_trn minibatches/epoch (and $n_vld validation mbs).")
    println("Evaluation is performed every $eval_nth minibatches (with $n_eval mbs).")
    println("Watch the progress with TensorBoard.")

    # mk log directory:
    #
    start_time = Dates.now()
    tb_log_dir = joinpath(tb_dir, tb_name,
                    Dates.format(start_time, "yyyy-mm-ddTHH:MM:SS"))
    # checkpoints:
    #
    cp_nth = Int(ceil(n_trn * cp_freq))

    # Tensorboard logger:
    #
    tbl = TensorBoardLogger.TBLogger(tb_log_dir,
                    min_level=Logging.Info)
    log_text(tbl, tb_log_dir, tb_name, start_time, tb_text)
    calc_and_report_loss_acc(mdl, takenth(trn, nth_trn),
            takenth(vld, nth_vld), tbl, 0)

    # Training:
    #
    mb_losses = Float32[]
    @showprogress for (i, mb_loss) in enumerate(opti(mdl, ncycle(trn,epochs)))

        push!(mb_losses, mb_loss)
        if (i % eval_nth) == 0
            calc_and_report_loss_acc(mdl, takenth(trn, nth_trn),
                    takenth(vld, nth_vld), tbl, eval_nth)
        end

        if (i % mb_loss_nth) == 0
            TensorBoardLogger.log_value(tbl,
                    "Minibatch loss (epoch = $n_trn steps)",
                    mean(mb_losses), step=i)
            mb_losses = Float32[]
        end

        if (i % cp_nth) == 0
            write_cp(mdl, i, tb_log_dir)
        end
    end
    return mdl
end


function log_text(tbl, tb_log_dir, tb_name, start_time, tb_text)

    tb_log_text =
    "<h2>NNHelferlein.jl tb_train!() log</h2>" *
    "   <ul> " *
    "   <li>dir:  $tb_log_dir</li> " *
    "   <li>name: $tb_name</li> " *
    "   <li>time: $(Dates.format(start_time, "E, yyyy/mm/dd, HH:MM:SS"))</li> " *
    "   </ul> " *
    "   <p> " *
    "   $tb_text " *
    "   </p> "


    TensorBoardLogger.log_text(tbl, "Description", tb_log_text, step=0)
    # with_logger(tbl) do
    #         @info "Description" text=tb_log_text log_step_increment=0
    # end
end

function write_cp(model, step, dir)

    dir_name = joinpath(dir, "checkpoints")
    if !isdir(dir_name)
        mkdir(dir_name)
    end
    fname = joinpath(dir_name, "checkpoint_$step.jld2")
    JLD2.@save fname model
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
            @info "Evaluation Loss" train=loss_trn valid=loss_vld log_step_increment=step
            @info "Evaluation Accuracy" train=acc_trn valid=acc_vld log_step_increment=0
    end
end



"""
    function predict_top5(mdl, x; top_n=5, classes=nothing)

Run the model mdl for data in x and prints the top 5
predictions as softmax probabilities.

### Arguments:
`top_n`: print top *n* hits instead of *5*
`classes` may be a list of human readable class labels.
"""
function predict_top5(mdl, x; top_n=5, classes=nothing)

    y = predict(mdl, x, softmax=false)

    if classes == nothing
        classes = repeat(["-"], maximum(top))
    end
    for (i,o) in enumerate(eachcol(y))
        o = Knet.softmax(vec(Array(o)))
        top = sortperm(vec(Array(o)), rev=true)[1:top_n]
        println("top-$top_n hits for sample $i: $top"); flush(stdout)

        @printf("%6s  %6s   %s\n", "softmax", "#", "class label")
        for t in top
            @printf(" %6.2f  %6i   \"%s\"\n", o[t], t, classes[t])
        end
        println(" ")
    end
    return y
end

"""
    function predict(mdl, x; softmax=false)

Return the prediction for x.

### Arguments:
`mdl`: executable network model
`x`: tensor, minibatch or iterator providing minibatches
        of input data
`softmax`: if true and if model is a `::Classifier` the prediction
        softmax probabilities are rezrned instead of raw
        activations.
"""
function predict(mdl, x; softmax=false)

    if x isa AbstractArray
        y = mdl(x)
    else
        # ys = [mdl(i) for i in x]
        y = cat((mdl(i) for i in x)..., dims=2)
    end
    y = convert(Array{Float32}, y)

    if softmax && mdl isa Classifier
        return Knet.softmax(y)
    else
        return y
    end
end
