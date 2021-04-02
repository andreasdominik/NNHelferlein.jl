"""
    function tb_train!(mdl, opti, trn, vld=nothing; epochs=1,
                      lr_decay=nothing, lrd_freq=1, l2=0.0,
                      eval_size=0.2, eval_freq=1,
                      acc_fun=nothing,
                      mb_loss_freq=100,
                      cp_freq=nothing, cp_dir="checkpoints",
                      tb_dir="logs", tb_name="run",
                      tb_text=\"\"\"Description of tb_train!() run.\"\"\",
                      opti_args...)

Train function with TensorBoard integration. TB logs are written with
the TensorBoardLogger.jl package.
The model is updated (in-place) and the trained model is returned.

### Arguments:
+ `mdl`: model; i.e. forward-function for the net
+ `opti`: Knet-stype optimiser type
+ `trn`: training data; iterator to provide (x,y)-tuples with
        minibatches
+ `vld`: validation data; iterator to provide (x,y)-tuples with
        minibatches. Set to `nothing`, if not defined.

### Keyword arguments:
#### Optimiser:
+ `epochs=1`: number of epochs to train
+ `lr_decay=nothing`: Leraning rate decay if not `nothing`:
        factor (<1) to reduce the
        lr every epoch as `lr  *= lr_decay`.
+ `lrd_freq=1`: frequency of learning rate decay steps. Default is
        to modify the lr after every epoch
+ `l2=0.0`: L2 regularisation; implemented as weight decay per
        parameter
+ `opti_args...`: optional keyword arguments for the optimiser can be specified
        (i.e. `lr`, `gamma`, ...).

#### Model evaluation:
+ `eval_size=0.2`: fraction of validation data to be used for calculating
        loss and accuracy for train and validation data during training.
+ `eval_freq=1`: frequency of evaluation; default=1 means evaluation is
        calculated after each epoch. With eval_freq=10 eveluation is
        calculated 10 times per epoch.
+ `acc_fun=nothing`: function to calculate accuracy. The function
        is called with 2 arguments: `fun(predictions, teaching)` where
        `predictions` is the output of a model call and a matrix and
        `teaching` is the teaching input (y).
        For classification tasks, `accuracy` from the Knet package is
        a good choice. For regression a correlation or mean error
        may be used (i.e. `acc_fun=(x,y)->sum(abs, x.-y)`).
+ `mb_loss_freq=100`: frequency of training loss reporting. default=100
        means that 100 loss-values per epoch will be logged to TensorBoard.
        If mb_loss_freq is greater then the number of minibatches,
        loss is logged for each minibatch.
+ `cp_freq=nothing`: frequency of model checkpoints written to disk.
        Default is `nothing`, i.e. no checkpoints are written.
        To write the model after each epoch with
        name `model` use freq=1; to write every 2 epochs freq=0.5.
+ `cp_dir="checkpoints"`: directory for checkpoints

#### TensorBoard:
TensorBoard log-directory is created from 3 parts:
`tb_dir/tb_name/<current date time>`.

+ `tb_dir="logs"`: root directory for tensorborad logs.
+ `tb_name="run"`: name of training run. `tb_name` will be used as
        directory name and should not include whitespace
+ `tb_text`:  description
        to be included in the TensorBoard log as *text* log.
"""
function tb_train!(mdl, opti, trn, vld=nothing; epochs=1,
                  lr_decay=nothing, lrd_freq=1, l2=0.0,
                  eval_size=0.2, eval_freq=1, acc_fun=nothing,
                  mb_loss_freq=100,
                  cp_freq=nothing, cp_dir="checkpoints",
                  tb_dir="logs", tb_name="run",
                  tb_text=""""Description of tb_train!() run.""",
                  opti_args...)

    # use every n-th mb for evaluation (based on vld if defined):
    #
    n_trn = length(trn)

    if vld == nothing
        n_vld = 0
        eval_vld = nothing
        n_eval = Int(ceil(n_trn * eval_size))
    else
        n_vld = length(vld)
        n_eval = Int(ceil(n_vld * eval_size))
        nth_vld = cld(n_vld, n_eval)
        eval_vld = takenth(vld, nth_vld)
    end
    nth_trn = cld(n_trn, n_eval)
    eval_trn = takenth(trn, nth_trn)

    # frequencies of actions during training:
    #
    eval_nth = cld(n_trn, eval_freq)
    mb_loss_nth = cld(n_trn, mb_loss_freq)
    lr_nth = cld(n_trn, lrd_freq)
    if cp_freq != nothing
        cp_nth = cld(n_trn, cp_freq)
    end


    # mk log directory:
    #
    start_time = Dates.now()
    tb_log_dir = joinpath(pwd(), tb_dir, tb_name,
                    Dates.format(start_time, "yyyy-mm-ddTHH-MM-SS"))
    println("Training $epochs epochs with $n_trn minibatches/epoch")
    if vld != nothing
        println("    (and $n_vld validation mbs).")
    end
    println("Evaluation is performed every $eval_nth minibatches (with $n_eval mbs).")
    println("Watch the progress with TensorBoard at: $tb_log_dir")


    # Tensorboard logger:
    #
    tbl = TensorBoardLogger.TBLogger(tb_log_dir,
                    min_level=Logging.Info)
    log_text(tbl, tb_log_dir, tb_name, start_time, tb_text,
             opti, trn, vld, epochs,
             lr_decay, lrd_freq, l2,
             cp_freq, opti_args)
    calc_and_report_loss(mdl, eval_trn, eval_vld, tbl, 0)

    if acc_fun != nothing
        calc_and_report_acc(mdl, acc_fun,eval_trn, eval_vld, tbl, 0)
    end


    # set optimiser:
    #
    for p in params(mdl)
        p.opt = opti(;opti_args...)
    end

    # Training:
    #
    mb_losses = Float32[]
    @showprogress for (i, (x,y)) in enumerate(ncycle(trn,epochs))

        loss = @diff mdl(x,y)
        mb_loss = value(loss)

        if isnan(mb_loss)
            println("ERROR: training aborted because of loss value NaN!")
            break
        end

        for p in params(loss)
            Δw = grad(loss, p) + p .* l2
            # println("updating $i: $(p.opt.lr), Δw: -")
            Knet.update!(p, Δw)
        end

        # TensorBoard:
        #
        # println("mb_loss: $mb_loss"); flush(stdout)
        push!(mb_losses, mb_loss)
        if (i % eval_nth) == 0
            calc_and_report_loss(mdl, eval_trn, eval_vld, tbl, eval_nth)

            if acc_fun != nothing
                calc_and_report_acc(mdl, acc_fun, eval_trn, eval_vld,
                                    tbl, eval_nth)
            end
        end

        if (i % mb_loss_nth) == 0
            TensorBoardLogger.log_value(tbl,
                    "Minibatch loss (epoch = $n_trn steps)",
                    mean(mb_losses), step=i)
            # println("mb_loss-mean: $(mean(mb_losses))"); flush(stdout)
            mb_losses = Float32[]
        end

        # checkpoints:
        #
        if (cp_freq != nothing) && (i % cp_nth) == 0
            write_cp(mdl, i, tb_log_dir)
        end

        # lr decay:
        #
        if (lr_decay != nothing) && (i % lr_nth == 0)
            lr = first(params(mdl)).opt.lr * lr_decay
            println("Set learning rate to η = $lr")
            for p in params(mdl)
                p.opt.lr = lr
                # println("adapting lr in $p to $(p.opt.lr)")
            end
        end
    end

    println("Training finished with:")
    println("Training loss:       $(calc_loss(mdl, data=trn))")
    if acc_fun != nothing
        println("Training accuracy:   $(calc_acc(mdl, acc_fun, data=trn))")
    end

    if vld != nothing
        println("Validation loss:     $(calc_loss(mdl, data=vld))")
        if acc_fun != nothing
            println("Validation accuracy: $(calc_acc(mdl, acc_fun, data=vld))")
        end
    end

    # save final model:
    #
    if (cp_freq != nothing)
        write_cp(mdl, n_trn*epochs+1, tb_log_dir)
    end
    return mdl
end


function log_text(tbl, tb_log_dir, tb_name, start_time, tb_text,
                  opti, trn, vld, epochs,
                  lr_decay, lrd_freq, l2,
                  cp_freq, opti_args)

    if vld == nothing
        vld = []
    end

    tb_log_text =
    "<h2>NNHelferlein.jl tb_train!() log</h2>" *
    "   <ul> " *
    "   <li>dir:  $tb_log_dir</li> " *
    "   <li>name: $tb_name</li> " *
    "   <li>time: $(Dates.format(start_time, "E, yyyy/mm/dd, HH:MM:SS"))</li> " *
    "   </ul> " *
    "   <p> " *
    "   $tb_text " *
    "   </p> " *
    "<h3>Training parameters:</h3>" *
    "   <ul>" *
    "   <li>Training minibatches: $(length(trn))</li>" *
    "   <li>Validation minibatches: $(length(vld))</li>" *
    "   <li>Optimiser: $opti</li>" *
    "   <li>Epochs: $epochs</li>"

    if lr_decay != nothing
        tb_log_text *= "   <li>lr-decay: $lr_decay with frequency $lrd_freq</li>"
    end
    if l2 > 0
        tb_log_text *= "   <li>L2 regularisation: $l2</li>"
    end
    if cp_freq != nothing
        tb_log_text *= "   <li>Checkpoints are saved $cp_freq times per epoch</li>"
    end

    for arg in keys(opti_args)
        tb_log_text *= "   <li>$arg: $(opti_args[arg])</li>"
    end
    tb_log_text *= "   </ul>"

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
    JLD2.@save fname, model
end

# Helper to calc loss and acc with only ONE forward run:
#
function calc_loss(mdl; data)

    loss = 0.0
    for (x,y) in data
        loss += mdl(x,y)
    end
    return loss/length(data)
end

function calc_acc(mdl, fun; data)

    acc = 0.0
    for (x,y) in data
        acc += fun(mdl(x), y)
    end
    return acc/length(data)
end


# Helper for TensorBoardLogger:
#
function calc_and_report_loss(mdl, trn, vld, tbl, step)

    loss_trn = calc_loss(mdl, data=trn)

    if vld != nothing
        loss_vld = calc_loss(mdl, data=vld)
        with_logger(tbl) do
            @info "Evaluation Loss" train=loss_trn valid=loss_vld log_step_increment=step
        end
    else
        with_logger(tbl) do
            @info "Evaluation Loss" train=loss_trn log_step_increment=step
        end
    end
end

function calc_and_report_acc(mdl, acc_fun, trn, vld, tbl, step)

    acc_trn = calc_acc(mdl, acc_fun, data=trn)

    if vld != nothing
        acc_vld = calc_acc(mdl, acc_fun, data=vld)
        with_logger(tbl) do
            @info "Evaluation Accuracy" train=acc_trn valid=acc_vld log_step_increment=step
        end
    else
        with_logger(tbl) do
            @info "Evaluation Accuracy" train=acc_trn log_step_increment=step
        end
    end
end



"""
    function predict_top5(mdl, x; top_n=5, classes=nothing)

Run the model `mdl` for data in `x` and print the top 5
predictions as softmax probabilities.

### Arguments:
+ `top_n`: print top *n* hits
+ `classes`: optional list of human readable class labels.
"""
function predict_top5(mdl, x; top_n=5, classes=nothing)

    y = predict(mdl, x, softmax=false)

    if classes == nothing
        classes = repeat(["-"], size(y)[1])
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
+ `mdl`: executable network model
+ `x`: iterator providing minibatches
        of input data
+ `softmax`: if true and if model is a `::Classifier` the predicted
        softmax probabilities are returned instead of raw
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
