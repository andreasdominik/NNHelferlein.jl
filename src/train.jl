"""
    function tb_train!(mdl, opti, trn, vld=nothing; epochs=1,
                      lr_decay=nothing, lrd_freq=1, lrd_linear=false,
                      l2=0.0,
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
+ `lr_decay=nothing`: do a leraning rate decay if not `nothing`:
        the value given is the final learning rate after `epochs*lrd_freq`
        steps of decay. `lr_decay` is only applied if both start learning rate
        `lr` and final learning rate `lr_decay` are defined explicitly.
        Example: `lr=0.01, lr_decay=0.001` will reduce the lr from
        0.01 to 0.001 during the training.
+ `lrd_freq=1`: frequency of learning rate decay steps. Default is
        to modify the lr after every epoch
+ `lrd_linear=false`: type of learning rate decay;
        If `false`, lr is modified
        by a constant factor (e.g. 0.9) resulting in an exponential decay.
        If `true`, lr is modified by the same step size, i.e. linearly.
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
        must implement the following signature: `fun(model; data)` where
        data is an iterator that provides (x,y)-tuples of minibatches.
        For classification tasks, `accuracy` from the Knet package is
        a good choice. For regression a correlation or mean error
        may be used.
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
                  lr_decay=nothing, lrd_freq=1, lrd_linear=false,
                  l2=0.0,
                  eval_size=0.2, eval_freq=1, acc_fun=nothing,
                  mb_loss_freq=100,
                  cp_freq=nothing, cp_dir="checkpoints",
                  tb_dir="logs", tb_name="run",
                  tb_text="""Description of tb_train!() run.""",
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


    # do lr-decay only if lr is explicitly defined:
    #
    if lr_decay != nothing && haskey(opti_args, :lr)
       lr_decay = calc_d_η(opti_args[:lr], lr_decay, lrd_linear, lrd_freq*epochs)
    else
        lr_decay = nothing
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
        calc_and_report_acc(mdl, acc_fun, eval_trn, eval_vld, tbl, 0)
    end


    # set optimiser - only if not yet set:
    #
    if first(params(mdl)).opt == nothing
        for p in params(mdl)
            p.opt = opti(;opti_args...)
        end
    else
        if haskey(opti_args, :lr)
            set_learning_rate(mdl, opti_args[:lr])
        end
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
            lr = first(params(mdl)).opt.lr
            lr = lrd_linear ? lr + lr_decay : lr * lr_decay
            @printf("Setting learning rate to η=%.2e\n", lr)
            set_learning_rate(mdl, lr)
            # for p in params(mdl)
            #     p.opt.lr = lr
            #     # println("adapting lr in $p to $(p.opt.lr)")
            # end
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


function set_learning_rate(mdl, lr)

    for p in params(mdl)
        p.opt.lr = lr
    end
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
        tb_log_text *= "   <li>learning rate step size: $lr_decay with frequency $lrd_freq</li>"
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
    JLD2.@save fname model
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

    # acc = 0.0
    # for (x,y) in data
    #     acc += fun(mdl(x), y)
    # end
    # return acc/length(data)
    return fun(mdl, data=data)
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
            @info "Evaluation Loss" train=loss_trn log_step_increment=0
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

# calc step size for lr-decay based on
# start rate, end rate and num steps:
#
function calc_d_η(η_start, η_end, lrd_linear, steps)
    if !lrd_linear
        d = log(η_end/η_start) / steps
        d = exp(d)
    else
        d = (η_end - η_start) / steps
    end
    return d
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


"""
    function hamming_dist(p, t; accuracy=false, vocab=nothing, pad=0)

Return the Hamming distance between two sequences or two minibatches
of sequences. Predicted sequences `p` and teaching input sequences `t`
may be of different length but the number of sequences in the minibatch
must be the same.

### Arguments:
+ `p`, `t`: n-dimensional arrays of type `Int` with predictions
        and teaching input for a minibatch of sequences.
        Shape of the arrays must be identical except of the first dimension
        (i.e. the sequence length) that may differ between `p` and `t`.
+ `accuracy=false`: if `false`, the mean Hamming distance in the minibatch
        is returned (i.e. the average number of differences in the sequences).
        If `true`, the accuracy is returned
        for all not padded positions in a range (0.0 - 1.0).
+ `vocab=nothing`: target laguage vocabulary of type `NNHelferlein.WordTokenizer`.
        If defined,
        the padding token of `vocab` is used to mask all control tokens in the
        sequences (i.e. '<start>, <end>, <unknwon>, <pad>').
+ `pad=0`: if `vocab` is undefined, `pad` is used to pad `p`, if the sequence
        length of `p` is smaller than the length of `t`.
"""
function hamming_dist(p, t; accuracy=false, vocab=nothing, pad=0)

    # make 2d matrix of sequences:
    #
    n_seq_t = size(t)[1]
    t = reshape(t, n_seq_t,:)

    n_seq_p = size(p)[1]
    p = reshape(p, n_seq_p,:)

    n_mb = size(t)[2]

    # make all control-tokens the same:
    #
    if vocab != nothing
        START = vocab("<start>")
        END = vocab("<end>")
        UNK = vocab("<unknown>")
        PAD = vocab("<pad>")

        t[t .== START] .= PAD
        t[t .== END] .= PAD
        t[t .== UNK] .= PAD
    else
        PAD = pad
    end

    # make n_seq of p the same as t:
    #
    if n_seq_p > n_seq_t
        p = p[1:n_seq_t,:]
    end
    while size(p)[1] < n_seq_t
        p = vcat(p, repeat([PAD], inner=(1,n_mb)))
    end

    # mask preds same as teaching and count all
    # mask positions:
    #
    p[t .== PAD] .= PAD
    num_pad = length(t[t .== PAD])
    return accuracy ? (sum(p .== t) - num_pad)/(length(t)-num_pad) : sum(p .!= t)/n_mb
end
