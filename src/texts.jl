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
a list of Strings is aassigned to a unique number.
The created object includes a list of words (`i2w`)(ordered by their numbers) and
a dictionary `w2i` with the words as keys.

### Constructor:

    function WordTokenizer(texts; len=nothing, add_ctls=true)

With arguments:
+ `texts`: `AbstractArray` or iterable collection of `AbstractArray`s to be
        analysed.
+ `len=nothing`: maximum number of different words in the vocabulary.
        Additional words in texts will be encoded as unknown. If `nothing`,
        all words of the texts are included.
+ `add_ctls=true`: if true, control words are added to the vocabulary
        (extending the maximum length by 4): `"<start>"`, `"<end>"`,
        `"<pad>"` and `"<unknown>"`. `"<unknown>"` will allways be encoded
        with the largest number in the vocab (i.e. `i2w[end]`).

### Signatures:

    function (t::WordTokenizer)(w::AbstractString)

Encode a word and return the corresponding number in the vocabulary or
the highest number (i.e. `"<unknown>"`) if the word is not in the vocabulary.


    function (t::WordTokenizer)(i::Int)

Decode a word by returning the word corresponding to `i` or
"unknown" if the number is out of range of the vocabulary.

### Examples:

```Julia
julia> vocab = WordTokenizer(["I love Julia", "They love Python"]);
Julia> vocab(4)
"Julia"

julia> vocab("love")
1

julia> vocab.(split("I love Julia"))
3-element Array{Int64,1}:
 2
 1
 4

julia> vocab.([3,1,4])
3-element Array{String,1}:
 "They"
 "love"
 "Julia

 julia> vocab.(split("I love Scala"))
3-element Array{Int64,1}:
 2
 1
 9

julia> vocab.([2,1,9])
3-element Array{String,1}:
 "I"
 "love"
 "<unknown>"
 ```
"""
mutable struct WordTokenizer
    len
    w2i
    i2w
end

function WordTokenizer(texts; len=nothing, add_ctls=true)

    if texts isa AbstractString
        texts = [texts]
    end

    words = []
    for t in texts
        # clean:
        #
        t = Unicode.normalize(t)
        t = replace(t, Regex("[.!?,;:\"\']") => " ")
        t = replace(t, Regex(" {2,}") => " ")
        t = replace(t, Regex("^ ") => "")
        t = replace(t, Regex(" \$") => "")

        # split:
        #
        append!(words, split(t, " "))
    end

    # make vocab:
    #
    counts = Dict{String, Int}()
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
    if len != nothing && len < length(counts)
        pairs = pairs[1:len]
    end

    # add control tokens:
    #
    if add_ctls
        append!(pairs, ["<start>" => 0,
                        "<end>" => 0,
                        "<pad>" => 0,
                        "<unknown>" => 0])
    end

    # make encode and decode data structures:
    #
    w2i = Dict{String, Int}()
    i2w = String[]

    for (i, w) in enumerate(pairs)
        push!(i2w, w[1])
        w2i[w[1]] = i
    end

    return WordTokenizer(length(w2i), w2i, i2w)
end

function (t::WordTokenizer)(i::Int)
    if i > t.len
        return t.i2w[end]
    else
        return t.i2w[i]
    end
end

function (t::WordTokenizer)(w::AbstractString)
    if haskey(t.w2i, w)
        return t.w2i[w]
    else
        return t.len
    end
end

# """
#     function mk_lines_minibatch(dir, batchsize; split=false, fr=0.2,
#                                 balanced=false, shuffle=true, train=true,
#                                 make_ascii=true)
#
# Return an iterable text-loader-object that provides
# minibatches of text lines, relative to dir.
# Class labels are defined by the top-level directory name.
#
# ### Arguments:
# + `dir`: base-directory of the test dataset. The first level of
#         sub-dirs are used as class names.
# + `batchsize`: size of minibatches.
#
# ### Keyword arguments:
# + `split`: return two iterators for training and validation
# + `fr`: split fraction
# + `balanced`: return balanced data (i.e. same number of instances
#         for all classes). Balancing is achieved via oversampling
# + `shuffle`: if true, shuffle the images everytime the iterator
#         restarts
# + `train`: if true, minibatches with (x,y) Tuples are provided,
#         if false only x (for prediction)
# + `make_ascii`: provide ASCII codes instead of Unicode characters.
# """
# function mk_lines_minibatch(dir, batchsize; split=false, fr=0.5,
#                             balanced=false, shuffle=true, train=true,
#                             make_ascii=true)
#
#     l_paths = get_files_list(dir)
#     # l_class_names = get_class_names(dir, i_paths)
#     # classes = unique(i_class_names)
#     # i_classes = [findall(x->x==c, classes)[1] for c in i_class_names]
#     l_lines, l_classes_names = readLines(l_paths)
#
#     if split                    # return train_loader, valid_loader
#         ((xvld,yvld),(xtrn,ytrn)) = do_split(i_paths, i_classes, at=fr)
#         if balanced
#             (xtrn,ytrn) = do_balance(xtrn, ytrn)
#             (xvld,yvld) = do_balance(xvld, yvld)
#         end
#         trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
#                             batchsize, shuffle, train,
#                             aug_pipl, pre_proc)
#         vld_loader = ImageLoader(dir, xvld, yvld, classes,
#                             batchsize, shuffle, train,
#                             aug_pipl, pre_proc)
#         return trn_loader, vld_loader
#     else
#         xtrn, ytrn = i_paths, i_classes
#         if balanced
#             (xtrn, ytrn) = do_balance(i_paths, i_classes)
#         end
#         trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
#                             batchsize, shuffle, train,
#                             aug_pipl, pre_proc)
#         return trn_loader
#     end
# end
#
#
#
# """
#     function get_class_labels(d::DataLoader)
#
# Extracts a list of class labels from a DataLoader.
# """
# function get_class_labels(dl::DataLoader)
#     return dl.classes
# end
#
# """
#     struct ImageLoader <: DataLoader
#         dir
#         i_paths
#         i_classes
#         classes
#         batchsize
#         shuffle
#         train
#         aug_pipl
#         pre_proc
#     end
#
# Iterable image loader.
# """
# mutable struct ImageLoader <: DataLoader
#     dir
#     i_paths
#     i_classes
#     classes
#     batchsize
#     shuffle
#     train
#     aug_pipl
#     pre_proc
# end
#
#
# # two iterate funs for ImageLoader
# #
# function Base.iterate(il::ImageLoader)
#
#     if il.shuffle
#         # idx = Random.randperm(length(il.i_paths))
#         # il.i_paths .= il.i_paths[idx]   # xv = @view x[idx] ??
#         # il.i_classes .= il.i_classes[idx]
#         il.i_paths, il.i_classes = do_shuffle(il.i_paths, il.i_classes)
#     end
#     state = 1
#     return iterate(il, state)
# end
#
# # state is the index of the next image after the currect batch.
# #
# function Base.iterate(il::ImageLoader, state)
#
#     # println("State: $state")
#     # check if empty:
#     #
#     if state > length(il.i_paths)
#         return nothing
#     end
#
#     # range for next minibatch:
#     #
#     n = length(il.i_paths)
#     mb_start = state
#     mb_size = mb_start + il.batchsize > n ? n-mb_start+1 : il.batchsize
#
#     return mk_image_mb(il, mb_start, mb_size), mb_start+il.batchsize
# end
#
#
# function mk_image_mb(il, mb_start, mb_size)
#
#     if mb_size < 1
#         return nothing
#     end
#
#     # nice way:
#     # is = mb_start:mb_start+mb_size
#     # mb_i = Float32.(cat(read_one_image.(is, il)..., dims=4))
#
#     i = mb_start
#     mb_i = Float32.(read_one_image(i, il))
#     mb_i = reshape(mb_i, size(mb_i)..., 1)
#     i += 1
#     while i < mb_start+mb_size
#         img = Float32.(read_one_image(i, il))
#         mb_i = Float32.(cat(mb_i, img, dims=4))
#         i += 1
#     end
#
#     mb_y = UInt8.(il.i_classes[mb_start:mb_start+mb_size-1])
#
#     if CUDA.functional()
#         mb_i = KnetArray(mb_i)
#     end
#
#     if il.train
#         return mb_i, mb_y
#     else
#         return mb_i
#     end
# end
#
#
#
# function read_one_image(i, il)
#
#     img = Images.load(il.i_paths[i])
#     img = Images.RGB.(img)
#
#     if il.aug_pipl isa Augmentor.ImmutablePipeline
#         img = Augmentor.augment(img, il.aug_pipl)
#     end
#
#     img = Float32.(permutedims(Images.channelview(img), (3,2,1)))
#
#     if il.pre_proc != nothing && il.pre_proc isa Function
#         img = il.pre_proc(img)
#     end
#     return(img)
# end
#
#
#
#
#
#
#
# function Base.length(il::ImageLoader)
#     n = length(il.i_paths) / il.batchsize
#     return ceil(Int, n)
# end
