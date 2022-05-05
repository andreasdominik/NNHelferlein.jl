"""
    function tb_train!(mdl, opti, trn, vld=nothing; epochs=1, split=nothing,
                      lr_decay=nothing, lrd_steps=5, lrd_linear=false,
                      l2=nothing, l1=nothing,
                      eval_size=0.2, eval_freq=1,
                      acc_fun=nothing,
                      mb_loss_freq=100,
                      checkpoints=nothing, cp_dir="checkpoints",
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
        the value given is the final learning rate after `lrd_steps`
        steps of decay (`lr_decay` may be bigger than `lr`; in this case
        the leraning rate is increased). 
        `lr_decay` is only applied if both start learning rate
        `lr` and final learning rate `lr_decay` are defined explicitly.
        Example: `lr=0.01, lr_decay=0.001` will reduce the lr from
        0.01 to 0.001 during the training (by default in 5 steps).
+ `lrd_steps=5`: number of learning rate decay steps. Default is `5`, i.e.
        modify the lr 4 times during the training (resulting in 5 different 
        learning rates).
+ `lrd_linear=false`: type of learning rate decay;
        If `false`, lr is modified
        by a constant factor (e.g. 0.9) resulting in an exponential decay.
        If `true`, lr is modified by the same step size, i.e. linearly.
+ `l1=nothing`: L1 regularisation; implemented as weight decay per
        parameter
+ `l2=nothing`: L2 regularisation; implemented as weight decay per
        parameter
+ `opti_args...`: optional keyword arguments for the optimiser can be specified
        (i.e. `lr`, `gamma`, ...).

#### Model evaluation:
+ `split=nothing`: if no validation data is specified and split is a 
        fraction (between 0.0 and 1.0), the training dataset is splitted at the
        specified point (e.g.: if `split=0.8`, 80% of the minibatches are used 
        for training and 20% for validation).
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
        may be preferred.
+ `mb_loss_freq=100`: frequency of training loss reporting. default=100
        means that 100 loss-values per epoch will be logged to TensorBoard.
        If mb_loss_freq is greater then the number of minibatches,
        loss is logged for each minibatch.
+ `checkpoints=nothing`: frequency of model checkpoints written to disk.
        Default is `nothing`, i.e. no checkpoints are written.
        To write the model after each epoch with
        name `model` use cp_epoch=1; to write every second epochs cp_epoch=2, 
        etc.
+ `cp_dir="checkpoints"`: directory for checkpoints

#### TensorBoard:
TensorBoard log-directory is created from 3 parts:
`tb_dir/tb_name/<current date time>`.

+ `tb_dir="logs"`: root directory for TensorBoard logs.
+ `tb_name="run"`: name of training run. `tb_name` will be used as
        directory name and should not include whitespace
+ `tb_text`:  description
        to be included in the TensorBoard log as *text* log.
"""
function tb_train!(mdl, opti, trn, vld=nothing; epochs=1,
                  split=nothing,
                  lr_decay=nothing, lrd_steps=5, lrd_linear=false,
                  l2=nothing, l1=nothing,
                  eval_size=0.2, eval_freq=1, acc_fun=nothing,
                  mb_loss_freq=100,
                  checkpoints=nothing, cp_dir="checkpoints",
                  tb_dir="logs", tb_name="run",
                  tb_text="""Description of tb_train!() run.""",
                  opti_args...)


    # split training data if split given:
    #
    if isnothing(vld) && !isnothing(split)
        trn, vld = split_minibatches(trn, split, shuffle=true)
        println("Splitting dataset for training ($(Int(round(split*100)))%) and validation ($(Int(round((1-split)*100)))%).")
    end

    # use every n-th mb for evaluation (based on vld if defined):
    #
    n_trn = length(trn)

    if isnothing(vld)
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

    # learning rate steps:
    #
    if lrd_steps > epochs * n_trn
        lrd_steps = epochs * n_trn
    end
    lr_nth = cld(epochs*n_trn, lrd_steps) 

    # check point rate:
    #
    if !isnothing(checkpoints)
        cp_nth = n_trn * checkpoints
    end


    # do lr-decay only if lr is explicitly defined:
    #
    if !isnothing(lr_decay) && haskey(opti_args, :lr)
       lr_decay = calc_d_η(opti_args[:lr], lr_decay, lrd_linear, lrd_steps)
    else
        lr_decay = nothing
    end


    # prepare l1/l2:
    #
    if !isnothing(l2)
        l2 = Float32(l2 / 2)
    end
    if !isnothing(l1)
        l1 = Float32(l1)
    end

    # mk log directory:
    #
    start_time = Dates.now()
    tb_log_dir = joinpath(pwd(), tb_dir, tb_name,
                    Dates.format(start_time, "yyyy-mm-ddTHH-MM-SS"))
    
    # echo log:
    #
    function echo_log()
        print("Training $epochs epochs with $n_trn minibatches/epoch")
        if !isnothing(vld)
            println(" and $n_vld validation mbs.")
        else
            println(".")
        end
        println("Evaluation is performed every $eval_nth minibatches with $n_eval mbs.")
        println("Watch the progress with TensorBoard at:")
        println(tb_log_dir)
        flush(stdout)
    end
    echo_log()


    # Tensorboard logger:
    #
    tbl = TensorBoardLogger.TBLogger(tb_log_dir,
                    min_level=Logging.Info)
    log_text(tbl, tb_log_dir, tb_name, start_time, tb_text,
             opti, trn, vld, epochs,
             lr_decay, lrd_steps, l2, l1,
             checkpoints, opti_args)
    calc_and_report_loss(mdl, eval_trn, eval_vld, tbl, 0)

    if !isnothing(acc_fun)
        calc_and_report_acc(mdl, acc_fun, eval_trn, eval_vld, tbl, 0)
    end


    # set optimiser - only if not yet set:
    #
    if isnothing(first(params(mdl)).opt)
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
            Δw = grad(loss, p) 
            Knet.update!(value(p), Δw, p.opt) 
            
            if !isnothing(l2)
                p.value .+= p.value .* l2
            end
            if !isnothing(l1)
                p.value .+= sign.(p.value) .* l1
            end
        end


        # TensorBoard:
        #
        # println("mb_loss: $mb_loss"); flush(stdout)
        push!(mb_losses, mb_loss)
        if (i % eval_nth) == 0
            calc_and_report_loss(mdl, eval_trn, eval_vld, tbl, eval_nth)

            if !isnothing(acc_fun)
                calc_and_report_acc(mdl, acc_fun, eval_trn, eval_vld,
                                    tbl, 0)
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
        if (!isnothing(checkpoints)) && (i % cp_nth) == 0
            write_cp(mdl, i, joinpath(tb_log_dir, cp_dir))
        end

        # lr decay:
        #
        if (!isnothing(lr_decay)) && i > 1 && ((i-1) % lr_nth == 0)
            lr = first(params(mdl)).opt.lr
            lr = lrd_linear ? lr + lr_decay : lr * lr_decay
            @printf("Setting learning rate to η=%.2e in epoch %.1f\n", lr, i/n_trn)
            set_learning_rate(mdl, lr)
        end
    end

    println("Training finished with:")
    println("Training loss:       $(calc_loss(mdl, data=trn))")
    if !isnothing(acc_fun)
        println("Training accuracy:   $(calc_acc(mdl, acc_fun, data=trn))")
    end

    if !isnothing(vld)
        println("Validation loss:     $(calc_loss(mdl, data=vld))")
        if !isnothing(acc_fun)
            println("Validation accuracy: $(calc_acc(mdl, acc_fun, data=vld))")
        end
    end

    # save final model:
    #
    if (!isnothing(checkpoints))
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
                  lr_decay, lrd_steps, l2, l1,
                  checkpoints, opti_args)

    if isnothing(vld)
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

    if !isnothing(lr_decay)
        tb_log_text *= "   <li>learning rate is reduced to $lr_decay in $lrd_steps steps.</li>"
    end
    if !isnothing(l2)
        tb_log_text *= "   <li>L2 regularisation: $l2</li>"
    end
    if !isnothing(l1)
        tb_log_text *= "   <li>L1 regularisation: $l1</li>"
    end
    if !isnothing(checkpoints)
        tb_log_text *= "   <li>Checkpoints are saved $checkpoints times per epoch</li>"
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

    if !isdir(dir)
        mkdir(dir)
    end
    fname = joinpath(dir, "checkpoint_$step.jld2")
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

    if !isnothing(vld)
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

    if !isnothing(vld)
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
        d = log(η_end/η_start) / (steps-1)
        d = exp(d)
    else
        d = (η_end - η_start) / (steps-1)
    end
    return d
end

"""
    function predict_top5(mdl; data, top_n=5, classes=nothing)

Run the model `mdl` for data in minibatches `data` and print the top 5
predictions as softmax probabilities.

### Arguments:
+ `top_n`: print top *n* hits
+ `classes`: optional list of human readable class labels.
"""
function predict_top5(mdl; data, top_n=5, classes=nothing)

    y = predict(mdl; data=data, softmax=false)

    if isnothing(classes)
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
    function predict(mdl; data, softmax=false)
    function predict(mdl, x; softmax=false )

Return the prediction for minibatches of data.     
The signature follows the standard call
`predict(model, data=xxx)`.     
The second signature predicts a single minibatch of data.

### Arguments:
+ `mdl`: executable network model
+ `data=iterator`: iterator providing minibatches
        of input data; if the minibatches include y-values 
        (i.e. teaching input), predictions *and* the y-values 
        will be returned. 
+ `data`: single Array of input data (i.e. input for one minibatch)
+ `softmax`: if true or if model is of type `Classifier` the predicted
        softmax probabilities are returned instead of raw
        activations.
"""
function predict(mdl; data, softmax=false)

    if first(data) isa Tuple
        py = [(mdl(x), y) for (x,y) in data]
        p = cat((p for (p,y) in py)..., dims=2)
        y = cat((y for (p,y) in py)..., dims=2)
    else
        p = cat((mdl(x) for x in data)..., dims=2)
    end
    p = convert(Array{Float32}, p)

    if softmax || mdl isa Classifier
        p = Knet.softmax(p, dims=1)
    end

    if first(data) isa Tuple
        return p, y
    else
        return p
    end
end

function predict(mdl, x::Array; softmax=false )
    
    println("absarray")
    p = mdl(x)
    if softmax || mdl isa Classifier
        p = Knet.softmax(p, dims=1)
    end
    return p
end

# TODO: add de_embed?

