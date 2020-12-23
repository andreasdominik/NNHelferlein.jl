# working with images ...
#
# (c) A. Dominik, 2020


"""
    function mk_image_minibatch(dir, batchsize; split=false, fr=0.2,
                                balanced=false, shuffle=true)

Return an iterable image-loader-object that provides
minibatches of path-names of image files, relative to dir.

### Arguments:
+ `dir`: base-directory of the image dataset. The first level of
        sub-dirs are used as class names.
+ `batchsize`: size of minibatches
+ `spilt`: return two iterators for training and validation
+ `fr`: split fraction
+ `balanced`: return balanced data (i.e. same number of instances
        for all classes). Balancing is achieved via oversampling
+ `shuffle`: if true, shuffle the images.
"""
function mk_image_minibatch(dir, batchsize; split=false, fr=0.2,
                            balanced=false, shuffle=true)

    i_paths = get_files_list(dir)
    i_class_names = get_class_names(dir, i_paths)
    classes = unique(i_class_names)
    i_classes = [findall(x->x==c, classes)[1] for c in i_class_names]

    train_loader = ImageLoader(dir, i_paths, i_classes, classes,
                               batchsize, shuffle)

    if split
        # return train_loader, valid_loader
    else
        return train_loader
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
    end

Iterable image loader.
"""
struct ImageLoader
    dir
    i_paths
    i_classes
    classes
    batchsize
    shuffle
end


# two iterate funs for ImageLoader
#
function iterate(il::ImageLoader)

    if shuffle
        idx = Random.randperm(length(il.i_paths))
        il.i_paths .= il.i_paths[idx]   # xv = @view x[idx] ??
        il.i_classes .= il.i_classes[idx]
    end
    state = 1
    return iterate(il, state)
end

# state is the index of the next image after the currect batch.
#
function iterate(il::ImageLoader, state)

    # check if empty:
    #
    if state > length(il.i_paths)
        return nothing
    end

    # range for next minibatch:
    #
    n = length(il.i_paths)
    mb_start = state
    mb_size = mb_start + il.batchsize > n ? n-mb_start : il.batchsize

    return mk_image_mb(il, mb_start, mb_size), mb_start+mb_size+1
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
