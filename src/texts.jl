# working with text data ...
#
# (c) A. Dominik, 2021

const TOKEN_START = 1
const TOKEN_END = 2
const TOKEN_PAD = 3
const TOKEN_UNKOWN = 4

"""
    mutable struct WordTokenizer
        len
        w2i
        i2w
    end

Create a word-based vocabulary: every unique word of a String or
a list of Strings is assigned to a unique number.
The created object includes a list of words (`i2w`, ordered by their numbers) and
a dictionary `w2i` with the words as keys.

The constants `TOKEN_START, TOKEN_END, TOKEN_PAD` and `TOKEN_UNKOWN`
are exported.

### Constructor:

    function WordTokenizer(texts; len=nothing, add_ctls=true)

With arguments:
+ `texts`: `AbstractArray` or iterable collection of `AbstractArray`s to be
        analysed.
+ `len=nothing`: maximum number of different words in the vocabulary.
        Additional words in texts will be encoded as unknown. If `nothing`,
        all words of the texts are included.
+ `add_ctls=true`: if true, control words are added in front of the vocabulary
        (extending the maximum length by 4): `"<start>"=>1`, `"<end>"=>2`,
        `"<pad>"=>3` and `"<unknown>"=>4`.

### Signatures:

    function (t::WordTokenizer)(w::T; split_words=false, add_ctls=false)
                                where {T <: AbstractString}

Encode a word and return the corresponding number in the vocabulary or
the highest number (i.e. `"<unknown>"`) if the word is not in the vocabulary.

The encode-signature accepts the keyword arguments `split_words` and
`add_ctls`. If `split_words==true`, the input is treated as a sentence
and splitted into single words and an array of integer with the encoded
sequence is returned. If `add_ctls==true` the sequence will be framed
by `<start>` and `<end>` tokens.


    function (t::WordTokenizer)(i::Integer)

Decode a word by returning the word corresponding to `i` or
"<unknown>" if the number is out of range of the vocabulary.


    function (t::WordTokenizer)(s::AbstractArray{T}; add_ctls=false)
                               where {T <: AbstractString}

Called with an Array of Strings the tokeniser splits the strings
into words and returns an Array of `Array{Integer}` with each of the
input strings represented by a sequence of Integer values.


    function (t::WordTokenizer)(seq::AbstractArray{T}; add_ctls=false)
                                     where {T <: Integer}

Called with an Array of Integer values a single string  is returned
with the decoded token-IDs as words (space-separated).

### Base Signatures:

        function length(t::WordTokenizer)

Return the length of the vocab.


### Examples:

    julia> vocab = WordTokenizer(["I love Julia", "They love Python"]);
    Julia> vocab(8)
    "Julia"

    julia> vocab("love")
    5

    julia> vocab.(split("I love Julia"))
    3-element Array{Int64,1}:
     5
     6
     8

    julia> vocab.i2w
    9-element Array{String,1}:
     "<start>"
     "<end>"
     "<pad>"
     "<unknown>"
     "love"
     "I"
     "They"
     "Julia"
     "Python"

    julia> vocab.w2i
    Dict{String,Int64} with 9 entries:
      "I"         => 6
      "<end>"     => 2
      "<pad>"     => 3
      "They"      => 7
      "Julia"     => 8
      "love"      => 5
      "Python"    => 9
      "<start>"   => 1
      "<unknown>" => 4

    julia> vocab.([7,5,8])
    3-element Array{String,1}:
     "They"
     "love"
     "Julia

    julia> vocab.("I love Scala", split_words=true)
    3-element Array{Int64,1}:
     6
     5
     4

    julia> vocab.([6,5,4])
    3-element Array{String,1}:
     "I"
     "love"
     "<unknown>"

    julia> vocab("I love Python", split_words=true, add_ctls=true)
    5-element Array{Int64,1}:
     1
     6
     5
     9
     2

    julia> vocab(["They love Julia", "I love Julia"])
    2-element Array{Array{Int64,1},1}:
     [7, 5, 8]
     [6, 5, 8]
"""
mutable struct WordTokenizer
    len
    w2i
    i2w
end


"""
    function clean_sentence(s)


Cleaning a sentence in some simple steps:
+ normalise Unicode
+ remove punctuation
+ remove duplicate spaces
+ strip
"""
function clean_sentence(s)

    s = Unicode.normalize(s)
    s = replace(s, Regex("[.!?,;:#~^\"\']") => " ")
    s = replace(s, Regex(" {2,}") => " ")
    s = strip(s)
    return s
end



function WordTokenizer(texts; len=nothing, add_ctls=true)

    if texts isa AbstractString
        texts = [texts]
    end

    words = []
    for t in texts
        t = clean_sentence(t)
        append!(words, split(t, " "))
    end

    # make vocab:
    #
    counts = Dict{String, Int32}()
    for w in words
        if haskey(counts, w)
            counts[w] += 1
        else
            counts[w] = 1
        end
    end
    pairs = sort(collect(counts), by=x->x[2], rev=true)

    # limit vocab length to len:
    #
    if !isnothing(len) && len < length(counts)
        pairs = pairs[1:len]
    end

    # add control tokens:
    #
    if add_ctls
        pushfirst!(pairs, "<start>" => 0,
                        "<end>" => 0,
                        "<pad>" => 0,
                        "<unknown>" => 0)
    end

    # make encode and decode data structures:
    #
    w2i = Dict{String, Int32}()
    i2w = String[]

    for (i, w) in enumerate(pairs)
        push!(i2w, w[1])
        w2i[w[1]] = Int32(i)
    end

    return WordTokenizer(length(w2i), w2i, i2w)
end

import Base.length
function Base.length(vocab::WordTokenizer)
    return vocab.len
end

function (t::WordTokenizer)(i::Integer)
    if i > t.len
        return t.i2w[end]
    else
        return t.i2w[i]
    end
end

function (t::WordTokenizer)(w::T; split_words=false,
                add_ctls=false) where {T <: AbstractString}

    w = clean_sentence(w)
    # tokenise a word or a complete string:
    #
    if !split_words
        if haskey(t.w2i, w)
            return t.w2i[w]
        else
            return t("<unknown>")
        end
    else
        s = split(w, " ")
        st = t.(s)
        if add_ctls
            st = vcat(t("<start>"), st, t("<end>"))
        end
        return st
    end
end

function (t::WordTokenizer)(s::AbstractArray{T}; o...) where {T <: AbstractString}

    # return a list of sequences:
    #
    return Array[t(w; split_words=true, o...) for w in s]
end

function (t::WordTokenizer)(seq::AbstractArray{T}; o...) where {T <: Integer}

    # return a list of sequences:
    #
    return join(t.(Array(seq), o...), " ")
end





"""
    function get_tatoeba_corpus(lang; force=false,
                url="https://www.manythings.org/anki/")

Download and read a bilingual text corpus from Tatoeba (provided)
by ManyThings (https://www.manythings.org).
All corpi are English-*Language*-pairs with different size and
quality. Considerable languages include:
+ `fra`: French-English, 180 000 sentences
+ `deu`: German-English, 227 000 sentences
+ `heb`: Hebrew-English, 126 000 sentences
+ `por`: Portuguese-English, 170 000 sentences
+ `tur`: Turkish-English, 514 000 sentences

The function returns two lists with corresponding sentences in both
languages. Sentences are *not* processed/normalised/cleaned, but
exactly as provided by Tatoeba.

The data is stored in the package directory and only downloaded once.

### Arguments:
+ `lang`: languagecode
+ `force=false`: if `true`, the corpus is downloaded even if
        a data file is already saved.
+ `url`: base url of ManyThings.
"""
function get_tatoeba_corpus(lang; force=false,
                url="https://www.manythings.org/anki/")


    @show dir = normpath(joinpath(dirname(pathof(@__MODULE__)),
                "..", "data", "Tatoeba"))
    if !ispath(dir)
        mkpath(dir)
    end
    fname = join([lang, "-eng.zip"])
    @show pathname = joinpath(dir, fname)

    # download if necessary:
    #
    if !isfile(pathname) || force
        url = join([url, fname])
        println("Downloading Tatoeba corpus for language $lang")
        println("from $url")
        download(url, pathname)
    else
        println("Corpus for language $lang is already downloaded.")
    end

    if !isfile(pathname)
        println("File $pathname not found!")
    end

    # read zipped file:
    #
    println("Reading Tatoeba corpus for languages en-$lang")
    z = ZipFile.Reader(pathname)
    en, lang = String[], String[]

    for f in z.files
        for (i,line) in enumerate(eachline(f))
            if i % 1000 == 0
                print("\rimporting sentences: $i")
            end
            splits = split(line, "\t")
            if length(splits) > 1
                enl, langl = splits[1:2]
                push!(en, enl)
                push!(lang, langl)
            end
        end
    end
    close(z)

    return en, lang
end


#################################################################################
# 
# deprecated seq minibatch!
#
# """
#     function seq_minibatch(x, [y,] batchsize; 
#                            seq_len=nothing, pad=3, o...)
# 
# Return an iterator of type `Knet.Data` with sequence minibatches from a
# list of sequences.
# All
# keyword args of [`Knet.minibatch()`](https://denizyuret.github.io/Knet.jl/latest/reference/#Knet.Train20.minibatch) can be used.
# 
# All sequences in x are brought to the same length by truncating (if too long)
# or padding with the token provided as `pad`.
# 
# If `y` is defined, the minibatches include the sequences for x and
# training targets `y`, given as n-dimensional array (as for `Knet.minibach()`).
# For sequence-2-sequence minibatches the function `seq2seq_minibatch()`
# must be used.
# 
# ### Arguments:
# + `x`: An iterable object of sequences.
# + `y`: vector or array with training targets
# + `batchsize`: size of minibatches
# + `seq_len=nothing`: demanded length of sequences in the minibatches.
#         If `nothing`, all sequences are padded to match with the longest
#         sequence.
# + `pad=3`: token, used for padding. The default (3) is the token set by
#         the `WordRTokenizer`. The token must be compatible
#         with the type of the sequence elements.
# + `o...`: any other keyword arguments of `Knet.minibatch()`, such as
#         `shuffle=true` or `partial=true` can be provided.
# """
# function seq_minibatch(x, y, batchsize; seq_len=nothing, pad=3, o...)
# 
#     if isnothing(seq_len)
#         seq_len = maximum(length.(x))
#     end
# 
#     x = pad_sequences(x, seq_len, pad)
#     return Knet.minibatch(x, y, batchsize; o...)
# end
# 
# 
# function seq_minibatch(x, batchsize; seq_len=nothing, pad=0, o...)
# 
#     if isnothing(seq_len)
#         seq_len = maximum(length.(x))
#     end
# 
#     x = pad_sequences(x, seq_len, pad)
#     return Knet.minibatch(x, batchsize; o...)
# end
# 
# 
# 
# """
#     function seq2seq_minibatch(x, y, batchsize; seq_len=nothing,
#                 pad_x=3, pad_y=x, o...)
# 
# Return an iterator of type `Knet.Data` with (x,y) sequence minibatches from
# two lists of sequences.
# All
# keyword args of [`Knet.minibatch()`](https://denizyuret.github.io/Knet.jl/latest/reference/#Knet.Train20.minibatch) can be used.
# 
# All sequences in x and y are brought to the same length
# by truncating (if too long)
# or padding with the token provided as `pad`.
# 
# ### Arguments:
# + `x`: An iterable object of sequences.
# + `y`: An iterable object of target sequences.
# + `batchsize`: size of minibatches
# + `seq_len=nothing`: demanded length of sequences in the minibatches.
#         If `nothing`, all sequences are padded to match with the longest
#         sequence. In case of `opti == true` sequences are truncated to 
#         `sqe_len`.
# + `optimize=false`: if `false` minibatches with the given seqence length are created. 
#         If `true` the sequence lengths are optimized to minimize padding, by sorting 
#         the sequences by their length and restricting the seq-length of each minibatch
#         to the longest sequence of the minibatch.
# + `pad_x=3`,
# + `pad_y=x`: token, used for padding. The token must be compatible
#         with the type of the sequence elements. If pad_y is omitted, pad_y is set 
#         equal to pad_x.
# + `o...`: any other keyword arguments of `Knet.minibatch()`, such as
#         `shuffle=true` or `partial=true` can be provided.
# """
# function seq2seq_minibatch(x, y, batchsize; seq_len=nothing, optimize=false,
#                            pad_x=3, pad_y=pad_x, o...)
# 
#     if optimize
#         return opti_minibatches(x,y, batch_size, seq_len, pad_x, pad_y, o...)
# 
#     else
#         if isnothing(seq_len)
#             seq_len = maximum((maximum(length.(x)), maximum(length.(y))))
#         end
# 
#         x = pad_sequences(x, seq_len, pad_x)
#         y = pad_sequences(y, seq_len, pad_y)
#     return Knet.minibatch(x, y, batchsize; o...)
#     end
# end
#
#
#################################################################################



"""
    function sequence_minibatch(x, [y], batchsize; 
                                pad=NNHelferlein.TOKEN_PAD, 
                                seq2seq=true, pad_y=pad,
                                x_padding=false,
                                shuffle=true, partial=false)


Return an iterator of type `DataLoader` with (x,y) sequence minibatches from
two lists of sequences.

All sequences within a minibatch in x and y are brought to the same length
by padding with the token provided as `pad`.

The sequences are sorted by length before building minibatches in order to 
reduce padding (i.e. sequences of similar length are combined to a minibatch).
If the same sequence length is needed for all minibatches, the sequences
must be truncated or padded before call of `sequence_minibatch()` 
(see functions `truncate_seqence()` and `pad_sequence()`).

### Arguments:
+ `x`: List of sequences of `Int`
+ `y`: List of sequences of `Int` or list of target values (i.e. teaching input)
+ `batchsize`: size of minibatches
+ `pad=NNHelferlein.PAD_TOKEN`,
+ `pad_y=x`: token, used for padding. The token must be compatible
        with the type of the sequence elements. If `pad_y` is omitted, it is set 
        equal to pad_x.
+ `seq2seq=true`: if `true` and `y` is provided, sequence-to-sequence minibatches are 
        created. Otherwise `y` is treated as scalar teaching input.
+ `shuffle=true`: The minibatches are shuffled as last step. If `false` the minibatches 
        with short sequences will be at the beginning of the dataset.
+ `partial=false`: If `true`, a partial minibatch will be created if necessaray to 
        include all input data.
+ `x_padding=false`: if `true`, pad sequences in x to make minibatches of the demanded size, 
        even if there are not
        enougth sequences of the same length in x.
        If `false`, partial minibatches are built (if partial == `true`) or remaining 
        sequneces are skipped (if partial == `false`).
"""
function sequence_minibatch(x, batchsize; 
                              pad=NNHelferlein.TOKEN_PAD, 
                              x_padding=false,
                              shuffle=false, partial=false)

    return sequence_minibatch(x, nothing, batchsize; 
                              seq2seq=false, 
                              pad=pad, x_padding=x_padding,
                              shuffle=shuffle, partial=partial)
end

function sequence_minibatch(x, y, batchsize; 
                              pad=NNHelferlein.TOKEN_PAD, 
                              seq2seq=true, pad_y=pad,
                              x_padding=false,
                              shuffle=false, partial=false)


    # sort seqs by length (of input):
    # only if shuffle!
    #
    if shuffle 
        idx = sortperm(length.(x))
    else
        idx = collect(1:length(x))
    end

    i = 1
    xmbs = []

    # set next mb from i to j:
    #
    while i <= length(x)
        if x_padding
            j = i+batchsize-1
            if j > length(x)
                j = length(x)
            end
        else # no padding for x_padding
            mb_seq_len = length(x[idx[i]])
            j = i
            while j+1 < i+batchsize && j+1 <= length(x) && length(x[idx[j+1]]) == mb_seq_len
                j += 1
            end
        end

        if j-i+1 == batchsize || partial
            push!(xmbs, one_mb(x, y, seq2seq, idx, i, j, pad, pad_y))
        end

        i = j + 1
    end
    return SequenceData(xmbs, shuffle=shuffle)
end

function one_mb(x, y, seq2seq, idx, i, j, pad, pad_y)

    xmb = mk_seq_mb(x[idx[i:j]], pad)
    if eltype(xmb) <: AbstractFloat
        xmb = convert2KnetArray(xmb)
    end

    if !isnothing(y)
    if seq2seq
            ymb = mk_seq_mb(y[idx[i:j]], pad_y)
            if eltype(ymb) <: AbstractFloat 
                ymb = convert2KnetArray(ymb)
            end
        else
            #ymb = y[idx[i:j]]
            ymb = cat(y[idx[i:j]]..., dims=ndims(y[i])+1)
            if eltype(ymb) <: AbstractFloat
                ymb = convert2KnetArray(ymb)
            end
        end
    end


    if isnothing(y)
       return xmb
    else
        return (xmb, ymb)
    end
end

function mk_seq_mb(x, pad)

    l = maximum(length.(x))
    x = pad_sequence.(x, l, token=pad)

    # return hcat(x...)
    return cat(x..., dims=ndims(x[1])+1)
end



"""
    function pad_sequence(s, len; token=NNHelferlein.TOKEN_PAD)

Stretch a sequence to length `len` by adding the padding token.
"""
function pad_sequence(s, len; token=NNHelferlein.TOKEN_PAD)

    s = deepcopy(s)
    if length(s) < len
        append!(s, repeat([token], len-length(s)))
    end
    return s
end

"""
    function truncate_sequence(s, len; end_token=nothing)

Truncate a sequence to the length `len`. 
If not `isnothing(end_token)`, the last token of the sequence is 
overwritten by the token.
"""
function truncate_sequence(s, len; end_token=nothing)

    s = deepcopy(s)
    # only do something if too long
    #
    if length(s) > len && !isempty(s)
        s = s[1:len]
    end

    # add <end> token if demanded:
    #
    if !isnothing(end_token) && !isempty(s)
        s[end] = end_token
    end
    return s
end







# """
#     function prepare_nlp_corpus(source, target; 
#                 batchsize=128, seq_len=14, 
#                 vocab_size=nothing, split=0.1)
# 
# Prepare a dataset for training from two lists of sentences in different
# languages.
# The function creates the vocabs and datasets of sequence-to-sequence minibatches
# for training and validation.
# Training data sentences are shuffled.
# 
# ### Arguments:
# + `source, target`: Lists of sentences (as Strings) for source and target 
#             language. List entries must correspond.
# + `batchsize`: Size of the minibatches.
# + `seq_len`: Sequence length. Short sequences are padded with the `<end>`-token;
#         long sequences are truncated.
# + `vocab_size`: Maximum size of vocab. If `nothing`, all words are included; i.e.
#             the vocab size equals teh number of different words in the datasets.
# + `split`: Ratio to split training and validation data.
# 
# ### Values:
# 
# 4 result objects are returned: source_vocab, target_vocab, train, valid.    
# 
# """
# function prepare_nlp_corpus(source, target; 
#             batchsize=128, seq_len=14, 
#             vocab_size=nothing, split=0.1)
# 
#     source = clean_sentence.(source)
#     target = clean_sentence.(target)
#     
#     src_vocab = WordTokenizer(source, len=vocab_size)
#     trg_vocab = WordTokenizer(target, len=vocab_size)
#     
#     src_seqs = src_vocab(source, add_ctls=true)
#     trg_seqs = trg_vocab(target, add_ctls=true)
#     
#     vld_num = Int(round(length(src_seqs) * split))
#     vld_ids = sort(sample(1:length(src_seqs), vld_num, replace=false))
#     
#     src_vld = src_seqs[vld_ids]
#     trg_vld = trg_seqs[vld_ids]
#     
#     src_trn = deleteat!(src_seqs, vld_ids)
#     trg_trn = deleteat!(trg_seqs, vld_ids)
#     
#     trn = seq2seq_minibatch(src_trn, trg_trn, batchsize, shuffle=true, seq_len=seq_len, 
#         pad_x=src_vocab("<end>"), pad_y=trg_vocab("<end>"))
#     vld = seq2seq_minibatch(src_vld, trg_vld, batchsize, shuffle=false, seq_len=seq_len, 
#         pad_x=src_vocab("<end>"), pad_y=trg_vocab("<end>"))
#     
#     return src_vocab, trg_vocab, trn, vld
# end 
