#
# funs for accuracy for different aplications
#
# (a)dominik, 2022
#


"""
    function peak_finder_acc(p, t; ret=:f1, verbose=0, 
                             tolerance=1, limit=0.5
    
    function peak_finder_acc(mdl; data=data, o...)

                        
Calculate an accuracy-like measure for data series consisting 
mainly of zeros and rare peaks.
The function counts the number of peaks in `y` detected by `p` 
(*true positives*), peaks not detected (*false negatives*) 
and the nnumber of peaks in `p` not present in `y` 
(*false positives*).

It is assumed that peaks in `y` are marked by a single value
higher as the limit (typically 1.0). Peaks in `p` may be 
broader; and are defined as local maxima with a value above
the limit.
If the tolerance ist set to >0, it may happen that the peaks at the first 
or last step are not evaluated (because evaluation stopss at 
`end-tolerance`).

If requested, *f1*, *G-mean* and *intersection over union* 
are calulated from the raw values .

### Arguments:
+ `p`, `t`: Predictions and teaching input (i.e. `y`) are 1-d series
            of data 
+ `ret`: return value as `Symbol`; one of 
        `:peaks`, `:recall`, `:precision`, `:miss_rate`, `:f1`,
        `:g_mean`, `:iou` or `:all`.
        If `:all` a named tuple is returned.
+ `vervose=0`: if `0`, no additional output is generated;
        if `1`, composite measures are printed to stdout;
        if `2`, all raw counts are printed.
+ `tolerance=1`: peak finder tolerance: The peak is defined as *correct*
        if it is detected within the tolerance.
+ `limit=0.5`: Only maxima with values above the limit are considered.
"""
function peak_finder_acc(p, t; ret=:f1, verbose=0, 
                               tolerance=1, limit=0.5)

    len = minimum((length(p), length(t)))
    peaks = 0
    tp = 0
    fp = 0
    fn = 0

    for i in 1+tolerance:len-tolerance

        # find FN:
        #
        if t[i] > limit  # peak in y
            peaks += 1
            if maximum(p[i-tolerance:i+tolerance]) >= limit  # peak in p
                tp += 1
            else
                fn += 1
            end
        end

        if p[i] > limit && p[i] > p[i-1] && p[i] > p[i+1] # local maximum > limit
            if maximum(t[i-tolerance:i+tolerance]) < limit
                fp += 1
            end
        end
    end

    recall = tp / ( tp+fn)
    precision = tp / (tp+fp)
    miss_rate = fn / (fn+tp)

    f1 = 2tp / (2tp + fp + fn)
    g_mean = √(precision * recall)
    iou = tp / (tp + fp + fn)
    
    r = (peaks = peaks,
        recall = recall,
        precision = precision,
        miss_rate = miss_rate,
        f1 = f1,
        g_mean = g_mean, 
        iou = iou)


    if verbose > 2
        println("Number of Peaks: $peaks")
        println("True Positives:  $tp")
        println("False Positives: $fp")
        println("False Negatives: $fn")
    end
    if verbose > 1
        println("Recall (TPR):    $recall")
        println("Precision (PPV): $precision")
        println("Miss Rate (FNR): $miss_rate")
        println("F1-Score:        $f1")
        println("G-Mean:          $g_mean")
        println("I over U:        $iou")
    end
    if verbose == 1 
        if ret == :g_mean
            println("G-Mean: $g_mean")
        elseif ret == :iou
            println("Intersection over Union: $iou")
        else
            println("F1-Score: $f1")
        end
    end

    if ret == :all
        return r
    else
        return r[ret]
    end
end

function peak_finder_acc(mdl; data=data, o...)

    acc = []
    i = 1
    for (x,y) in data
        println("running $i") ; flush(stdout); i += 1
        p = mdl(x)
        push!(acc, peak_finder_acc(p, y; o...))
    end
    return mean(acc)
end
    



"""
    function hamming_dist(p, t; accuracy=false, 
                                ignore_ctls=false, vocab=nothing, 
                                start=nothing, stop=nothing, pad=nothing, unk=nothing)


    function hamming_acc(p, t; o...)

    function hamming_acc(mdl; data=data, o...)

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
+ `ignore_ctls=false`: a vocab is used to replace all '<start>, <end>, <unknwon>, <pad>'
        tokens by `<pad>`. If true, padding and other control tokens are treated as
        normal codes and are not ignored.
+ `vocab=nothing`: target laguage vocabulary of type `NNHelferlein.WordTokenizer`.
        If defined,
        the padding token of `vocab` is used to mask all control tokens in the
        sequences (i.e. '<start>, <end>, <unknwon>, <pad>').
+ `start, stop, pad, unk`: may be used to define individual control tokens.
        default is `nothing`.

### Details:
The function `hamming_acc()` is a shortcut to return the accuracy instead of
the distance. The signature `hamming_acc(mdl; data=data; o...)` is for compatibility
with acc functions called by train.



"""
function hamming_dist(p, t; accuracy=false, ignore_ctls=false, vocab=nothing, 
                            start=nothing, stop=nothing, pad=nothing, unk=nothing)

    # make 2d matrix of sequences:
    #
    n_seq_t = size(t)[1]
    t = reshape(copy(t), n_seq_t,:)

    n_seq_p = size(p)[1]
    p = reshape(copy(p), n_seq_p,:)

    n_mb = size(t)[2]

    # make all control-tokens the same:
    #
    if !ignore_ctls
        if isnothing(vocab)   # use defaults
            PAD = 3
            START = 1
            END = 2
            UNK = 4
        else                  # use vocab
            START = vocab("<start>")
            END = vocab("<end>")
            UNK = vocab("<unknown>")
            PAD = vocab("<pad>")
        end

        if !isnothing(start)
            START = start
        end
        if !isnothing(stop)
            END = stop
        end
        if !isnothing(pad)
            PAD = pad
        end
        if !isnothing(unk)
            UNK = unk
        end

        t[t .== START] .= PAD
        t[t .== END] .= PAD
        t[t .== UNK] .= PAD
    end

    # make seqs the same length and 
    # add the rest to dist:
    #
    dist = 0
    if n_seq_p > n_seq_t
        p = p[1:n_seq_t,:]
        dist += n_seq_p - n_seq_t 
    elseif n_seq_t > n_seq_p
        t = t[1:n_seq_p,:]
        dist += n_seq_t - n_seq_p 
    end

    # mask preds same as teaching and count all
    # mask positions:
    #
    if ignore_ctls
        num_non_pad = length(t)
    else
        p[t .== PAD] .= PAD
        num_non_pad = length(t[t .!== PAD])
    end

    dist += sum(p .!= t)


    if accuracy
        correct = num_non_pad - dist
        if correct < 0
            correct = 0
        end
        return correct / num_non_pad

    else
        return dist/n_mb
    end

    # return accuracy ? (sum(p .== t) - num_pad)/(length(t)-num_pad) : sum(p .!= t)/n_mb
    return dist
end

function hamming_acc(p, t; o...)

    return hamming_dist(p, t; accuracy=true, o...)
end


function hamming_acc(mdl; data=data, o...)

    acc = []
    for (x,y) in data
        p = mdl(x)
        push!(acc, hamming_acc(p, y; o...))
    end
    return mean(acc)
end
    
