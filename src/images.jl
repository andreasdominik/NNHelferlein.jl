# working with images ...
#
# (c) A. Dominik, 2020

"""
    function mk_image_minibatch(dir, batchsize; split=false, fr=0.2,
                                balanced=false, shuffle=true, train=true,
                                pre_proc=nothing)

Return an iterable image-loader-object that provides
minibatches of path-names of image files, relative to dir.

### Arguments:
+ `dir`: base-directory of the image dataset. The first level of
        sub-dirs are used as class names.
+ `batchsize`: size of minibatches

### Keyword arguments:
+ `split`: return two iterators for training and validation
+ `fr`: split fraction
+ `balanced`: return balanced data (i.e. same number of instances
        for all classes). Balancing is achieved via oversampling
+ `shuffle`: if true, shuffle the images
+ `train`: if true, minibatches with (x,y) Tuples are provided,
        if false only x (for prediction)
+ `aug_pipl`: augmentation pipeline for Augmentor.jl. Augmentation
        is performed before the pre_proc-function is applied
+ `pre_proc`: function or list of functions with preprocessing
        and augmentation algoritms of type x = f(x).
"""
function mk_image_minibatch(dir, batchsize; split=false, fr=0.5,
                            balanced=false, shuffle=true, train=true,
                            aug_pipl=nothing, pre_proc=nothing)

    i_paths = get_files_list(dir)
    i_class_names = get_class_names(dir, i_paths)
    classes = unique(i_class_names)
    i_classes = [findall(x->x==c, classes)[1] for c in i_class_names]


    if split                    # return train_loader, valid_loader
        ((xtrn,ytrn), (xvld,yvld)) = do_split(i_paths, i_classes, at=fr)
        if balanced
            (xtrn,ytrn) = do_balance(xtrn, ytrn)
            (xvld,yvld) = do_balance(xvld, yvld)
        end
        trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc)
        vld_loader = ImageLoader(dir, xvld, yvld, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc)
        return trn_loader, vld_loader
    else
        xtrn, ytrn = i_paths, i_classes
        if balanced
            (xtrn, ytrn) = do_balance(i_paths, i_classes)
        end
        trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc)
        return trn_loader
    end
end


"""
    struct ImageLoader
        dir
        i_paths
        i_classes
        classes
        batchsize
        shuffle
        train
        aug_pipl
        pre_proc
    end

Iterable image loader.
"""
mutable struct ImageLoader
    dir
    i_paths
    i_classes
    classes
    batchsize
    shuffle
    train
    aug_pipl
    pre_proc
end


# two iterate funs for ImageLoader
#
function Base.iterate(il::ImageLoader)

    if il.shuffle
        # idx = Random.randperm(length(il.i_paths))
        # il.i_paths .= il.i_paths[idx]   # xv = @view x[idx] ??
        # il.i_classes .= il.i_classes[idx]
        il.i_paths, il.i_classes = do_shuffle(il.i_paths, il.i_classes)
    end
    state = 1
    return iterate(il, state)
end

# state is the index of the next image after the currect batch.
#
function Base.iterate(il::ImageLoader, state)

    println("State: $state")
    # check if empty:
    #
    if state > length(il.i_paths)
        return nothing
    end

    # range for next minibatch:
    #
    n = length(il.i_paths)
    mb_start = state
    mb_size = mb_start + il.batchsize > n ? n-mb_start+1 : il.batchsize

    return mk_image_mb(il, mb_start, mb_size), mb_start+il.batchsize
end


function mk_image_mb(il, mb_start, mb_size)

    if mb_size < 1
        return nothing
    end

    # nice way:
    # is = mb_start:mb_start+mb_size
    # mb_i = Float32.(cat(read_one_image.(is, il)..., dims=4))

    i = mb_start
    mb_i = Float32.(read_one_image(i, il))
    mb_i = reshape(mb_i, size(mb_i)..., 1)
    i += 1
    while i < mb_start+mb_size
        img = Float32.(read_one_image(i, il))
        mb_i = Float32.(cat(mb_i, img, dims=4))
        i += 1
    end

    mb_y = UInt8.(il.i_classes[mb_start:mb_start+mb_size-1])

    if CUDA.functional()
        mb_i = KnetArray(mb_i)
    end

    if il.train
        return mb_i, mb_y
    else
        return mb_i
    end
end



function read_one_image(i, il)

    img = Images.load(il.i_paths[i])

    if il.aug_pipl isa Augmentor.ImmutablePipeline
        img = Augmentor.augment(img, il.aug_pipl)
    end
    if il.pre_proc isa Function
        img = il.pre_proc(img)
    end
    img = Float32.(permutedims(Images.channelview(img), (3,2,1)))
    return(img)
end







function Base.length(il::ImageLoader)
    n = length(il.i_paths) / il.batchsize
    return ceil(Int, n)
end





# return a list of all files realtive to dir:
#
function get_files_list(dir)

    image_paths = String[]
    for (root, dirs, files) in walkdir(dir)
        if !isempty(files)
            append!(image_paths, joinpath.(root, files))
        end
    end
    # filter out non-images:
    #
    fi = r"(\.png$)|(\.jpg$)|(\.jpeg$)|(\.gif$)"i
    image_paths = filter(i->occursin(fi, i), image_paths)
    return image_paths
end


# extract class names from top-level dir of list of filenames:
#
function get_class_names(dir, image_paths)

    classes = String[]
    regex = Regex("$dir/?([^/]+)/")
    for image in image_paths
        m = match(regex, image)
        if m != nothing
            push!(classes, m[1])
        else
            push!(classes, "unknown_class")
        end
    end

    return classes
end
