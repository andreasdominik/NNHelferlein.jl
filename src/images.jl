# working with images ...
#
# (c) A. Dominik, 2020

const UNKNOWN_CLASS = "unknwon_class"

"""
    function mk_image_minibatch(dir, batchsize; split=false, fr=0.2,
                                balanced=false, shuffle=true, train=true,
                                pre_load=false,
                                aug_pipl=nothing, pre_proc=nothing)

Return one or two iterable image-loader-objects that provides
minibatches of images. For training each minibatch is a tupel
`(x,y)` with x: 4-d-array with the minibatch of data and y:
vector of class IDs as Int.

### Arguments:
+ `dir`: base-directory of the image dataset. The first level of
        sub-dirs are used as class names.
+ `batchsize`: size of minibatches

### Keyword arguments:
+ `split`: return two iterators for training and validation
+ `fr`: split fraction
+ `balanced`: return balanced data (i.e. same number of instances
        for all classes). Balancing is achieved via oversampling
+ `shuffle`: if true, shuffle the images everytime the iterator
        restarts
+ `train`: if true, minibatches with (x,y) Tuples are provided,
        if false only x (for prediction)
+ `pre_load`: if `true` all images are loaded in advance;
        otherwise images are loaded on demand durng training.
        (option is *not implemented yet!*)
+ `aug_pipl`: augmentation pipeline for Augmentor.jl. Augmentation
        is performed before the pre_proc-function is applied
+ `pre_proc`: function with preprocessing
        and augmentation algoritms of type x = f(x). In contrast
        to the augmentation that modifies images, is `pre_proc`
        working on Arrays{Float32}.
+ `pre_load=false`: read all images from disk once when populating the
        loader (requires loads of memory, but speeds up training).
"""
function mk_image_minibatch(dir, batchsize; split=false, fr=0.5,
                            balanced=false, shuffle=true, train=true,
                            pre_load=false,
                            aug_pipl=nothing, pre_proc=nothing)

    i_paths = get_files_list(dir)
    i_n = length(i_paths)

    if train
        i_class_names = get_class_names(dir, i_paths)
        classes = unique(i_class_names)
        i_classes = [findall(x->x==c, classes)[1] for c in i_class_names]
    else
        i_class_names = fill(UNKNOWN_CLASS, i_n)
        classes =[UNKNOWN_CLASS]
        i_classes = fill(0, i_n)
    end


    if split                    # return train_loader, valid_loader
        ((xvld,yvld),(xtrn,ytrn)) = do_split(i_paths, i_classes, at=fr)
        if balanced
            (xtrn,ytrn) = do_balance(xtrn, ytrn)
            (xvld,yvld) = do_balance(xvld, yvld)
        end
        trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc,
                            pre_load)

        vld_loader = ImageLoader(dir, xvld, yvld, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc,
                            pre_load)
        return trn_loader, vld_loader
    else
        xtrn, ytrn = i_paths, i_classes
        if balanced
            (xtrn, ytrn) = do_balance(i_paths, i_classes)
        end
        trn_loader = ImageLoader(dir, xtrn, ytrn, classes,
                            batchsize, shuffle, train,
                            aug_pipl, pre_proc,
                            pre_load)
        return trn_loader
    end
end



"""
    function get_class_labels(d::DataLoader)

Extracts a list of class labels from a DataLoader.
"""
function get_class_labels(dl::DataLoader)
    return dl.classes
end

"""
    struct ImageLoader <: DataLoader
        dir
        i_paths
        i_classes
        classes
        batchsize
        shuffle
        train
        aug_pipl
        pre_proc
        pre_load
        i_images
    end

Iterable image loader to provide minibatches of images as
4-d-arrays (x,y,rgb,mb).
"""
mutable struct ImageLoader <: DataLoader
    dir                 # root dir
    i_paths             # list of all file image_paths
    i_classes           # list of classs IDs for each image
    classes             # unique list of class names
    batchsize           #
    shuffle             # if true: shuffle for each start
    i_sequence          # actual sequrence of images to take
    train               # if true: inlcude y in minibatches
    aug_pipl            # Augmentor.jl pipeline
    pre_proc            # function to process one 3d image tensor (RGB)
    pre_load            # load all images on init
    i_images            # list of all images; nothing if not predolad.

    function ImageLoader(dir, i_paths, i_classes, classes,
                         batchsize, shuffle, train,
                         aug_pipl, pre_proc, pre_load)

        i_seq = collect(1:length(i_paths))
        if pre_load
            i_images = pre_load_images(i_paths)
        else
            i_images = nothing
        end
        return new(dir, i_paths, i_classes, classes,
                    batchsize, shuffle, i_seq,
                    train,
                    aug_pipl, pre_proc,
                    pre_load, i_images)
    end
end


# two iterate funs for ImageLoader
#
function Base.iterate(il::ImageLoader)

    if il.shuffle
        Random.shuffle!(il.i_sequence)
    end
    state = 1
    return iterate(il, state)
end

# state is the index of the next image after the currect batch.
#
function Base.iterate(il::ImageLoader, state)

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
    state = mb_start+il.batchsize

    return mk_image_mb(il, mb_start, mb_size), state
end


function mk_image_mb(il, mb_start, mb_size)

    if mb_size < 1
        return nothing
    end

    # nice way:
    image_ns = mb_start:mb_start+mb_size-1

    # make image index form shuffled list entry:
    #
    image_ns = il.i_sequence[image_ns]

    # avoid broadcasting of iterator il:
    #
    mb_i = cat(read_one_image.(image_ns, Ref(il))..., dims=4)
    mb_y = UInt8.(il.i_classes[image_ns])

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

    if il.pre_load
        img = il.i_images[i]
    else
        img = Images.RGB.(Images.load(il.i_paths[i]))
    end

    if il.aug_pipl isa Augmentor.ImmutablePipeline ||
        il.aug_pipl isa Augmentor.ImageOperation
        img = Augmentor.augment(img, il.aug_pipl)
    end

    if isa(img[1,1], Colors.RGB)
        img = Float32.(permutedims(Images.channelview(img), (3,2,1)))
    else
        img = Float32.(Images.channelview(img))
    end

    if il.pre_proc != nothing && il.pre_proc isa Function
        img = il.pre_proc(img)
    end
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
    for image_path in image_paths
        image_path = replace(image_path, dir=>"")
        dirs = splitpath(image_path)
        while length(dirs) > 0 && !occursin(r"[A-Za-z1-9]", dirs[1])
            popfirst!(dirs)
        end
        if length(dirs) < 2
            class = UNKNOWN_CLASS
        else
            class = dirs[1]
        end
        push!(classes, class)
    end

    return classes
end


# read all images into mem:
#
function pre_load_images(i_paths)

    images = []
    for path in i_paths
        push!(images, Images.RGB.(Images.load(path)))
    end
    return images
end



"""
    function image2array(img)

Take an image and return a 3d-array for RGB and a 2d-array for grayscale
images with the colour channels as last dimension.
"""
function image2array(img)

    ch = Images.channelview(img)

    if length(size(ch)) == 2
     ch = reshape(ch, :,size(ch)...)
    end

    arr = Float32.(permutedims(ch, (3,2,1)))

    if CUDA.functional()
        arr = KnetArray(arr)
    end

    return arr
end


"""
    function array2image(arr)

Take a 3d-array with colour channels as last dimension or a 2d-array
and return an array of RGB or of Gray as Image.
"""
function array2image(arr)

    if length(size(arr)) == 2
        arr = permutedims(arr, (2,1))
        itype = Images.Gray

    elseif length(size(arr)) == 3
        arr = permutedims(arr, (3,2,1))
        if size(arr)[1] == 1
            arr = reshape(arr, size(arr)[2], size(arr)[3])
            itype = Images.Gray
        else
            itype = Images.RGB
        end
    end
    return Images.colorview(itype, Array(arr))
end


"""
    function array2RGB(arr)

Take a 3d-array with colour channels as last dimension or a 2d-array
and return always an array of RGB as Image.
"""
function array2RGB(arr)

    img = array2image(arr)
    return Images.RGB.(img)
end
