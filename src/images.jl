# working with images ...
#
# (c) A. Dominik, 2020


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

# extract classes from top-level dir:
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
