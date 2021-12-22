# working with text data ...
#
# (c) A. Dominik, 2021

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


    function (t::WordTokenizer)(i::Int)

Decode a word by returning the word corresponding to `i` or
"<unknown>" if the number is out of range of the vocabulary.


    function (t::WordTokenizer)(s::AbstractArray{T}; add_ctls=false)
                               where {T <: AbstractString}

Called with an Array of Strings the tokeniser splits the strings
into words and returns an Array of `Array{Int}` with each of the
input strings represented by a sequence of Integer values.


    function (t::WordTokenizer)(seq::AbstractArray{T}; add_ctls=false)
                                     where {T <: Int}

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
    if len !== nothing && len < length(counts)
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

function (t::WordTokenizer)(i::Int)
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

function (t::WordTokenizer)(seq::AbstractArray{T}; o...) where {T <: Int}

    # return a list of sequences:
    #
    return join(t.(seq, o...), " ")
end





"""
    function get_tatoeba_corpus(lang; force=false,
                url="https://www.manythings.org/anki/")

Download and read a bilingual text corpus from Tatoeba (privided)
by ManyThings (https://www.manythings.org).
All corpi are English-*Language*-pairs with different size and
quality. Considerable languages include:
+ `fra`: French-English, 180 000 sentences
+ `deu`: German-English, 227 000 sentences
+ `heb`: Hebrew-English, 126 000 sentences
+ `por`: Portuguese-English, 170 000 sentences
+ `tur`: Turkish-English, 514 000 sentences

The function returns two lists with corresponding sentences in both
languages. Sentences are are *not* processed/normalised/cleaned, but
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



"""
    function seq_minibatch(x, [y,] batchsize; seq_len=nothing, pad=3, o...)

Return an iterator of type `Knet.Data` with sequence minibatches from a
list of sequences.
all
keyword args of [`Knet.minibatch()`](https://denizyuret.github.io/Knet.jl/latest/reference/#Knet.Train20.minibatch) can be used.

All sequences in x are brought to the same length by truncating (if too long)
or padding with the token provided as `pad`.

If `y` is defined, the minibatches include the sequences for x and
training targets `y`, given as n-dimensional array (as for `Knet.minibach()`).
For sequence-2-sequence minibatches the function `seq2seq_minibatch()`
must be used.

### Arguments:
+ `x`: An iterable object of sequences.
+ `y`: vector or array with training targets
+ `batchsize`: size of minibatches
+ `seq_len=nothing`: demanded length of sequences in the minibatches.
        If `nothing`, all sequences are padded to match with the longest
        sequence.
+ `pad=3`: token, used for padding. The default (3) is the token set by
        the `WordRTokenizer`. The token must be compatible
        with the type of the sequence elements.
+ `o...`: any other keyword arguments of `Knet.minibatch()`, such as
        `shuffle=true` or `partial=true` can be provided.
"""
function seq_minibatch(x, y, batchsize; seq_len=nothing, pad=3, o...)

    if seq_len === nothing
        seq_len = maximum(length.(x))
    end

    x = pad_sequences(x, seq_len, pad)
    return Knet.minibatch(x, y, batchsize; o...)
end


function seq_minibatch(x, batchsize; seq_len=nothing, pad=0, o...)

    if seq_len === nothing
        seq_len = maximum(length.(x))
    end

    x = pad_sequences(x, seq_len, pad)
    return Knet.minibatch(x, batchsize; o...)
end


"""
    function seq2seq_minibatch(x, y, batchsize; seq_len=nothing,
                pad_x=3, pad_y=3, o...)

Return an iterator of type `Knet.Data` with (x,y) sequence minibatches from
two lists of sequences.
all
keyword args of [`Knet.minibatch()`](https://denizyuret.github.io/Knet.jl/latest/reference/#Knet.Train20.minibatch) can be used.

All sequences in x and y are brought to the same length
by truncating (if too long)
or padding with the token provided as `pad`.

### Arguments:
+ `x`: An iterable object of sequences.
+ `y`: An iterable object of target sequences.
+ `batchsize`: size of minibatches
+ `seq_len=nothing`: demanded length of sequences in the minibatches.
        If `nothing`, all sequences are padded to match with the longest
        sequence.
+ `pad_x=3`,
+ `pad_y=3`: token, used for padding. The token must be compatible
        with the type of the sequence elements.
+ `o...`: any other keyword arguments of `Knet.minibatch()`, such as
        `shuffle=true` or `partial=true` can be provided.
"""
function seq2seq_minibatch(x, y, batchsize; seq_len=nothing,
                           pad_x=3, pad_y=3, o...)

    if seq_len === nothing
        seq_len = maximum((maximum(length.(x)), maximum(length.(y))))
    end

    x = pad_sequences(x, seq_len, pad_x)
    y = pad_sequences(y, seq_len, pad_y)

    return Knet.minibatch(x, y, batchsize; o...)
end

function pad_sequences(s, len, pad)

    elem_type = typeof(s[1][1])
    data = Array{elem_type}(undef, len, length(s))

    for (i,seq) in enumerate(s)
        if length(seq) > len        # if too long
            seq = seq[1:len]
        end
        while length(seq) < len  # if too short
            push!(seq, pad)
        end
        data[:,i] = seq
    end
    return data
end
