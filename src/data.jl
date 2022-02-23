# funs for handling the example data, not handled by default GIT
#
const GITHUB_NNH = "https://github.com/andreasdominik/NNHelferlein.jl"
const GITHUB_DATA = GITHUB_NNH * "/raw/main/data"


"""
    function download_example_data(path_name..., force=false)

Download example data file from github large file storage.
All elements of the path (starting with the subdirectory in `/data` 
and ending with the filename)
must be provided as list or as arguments.

The file is only downloaded, if not already there 
(unless `force=true`).

### Examples:
```julia
julia> download_example_data("pretrained", "vgg16.h5")
```
"""
function download_example_data(path_name...; force=false)

    if path_name isa AbstractArray || path_name isa Tuple
        url = GITHUB_DATA * "/" * join(path_name, "/")
        pname = joinpath(NNHelferlein.DATA_DIR, path_name[1:end-1]...)
        fonly = path_name[end]
    else
        url = GITHUB_DATA * "/" * path_name
        pname = NNHelferlein.DATA_DIR
        fonly = path_name
    end
    fname = joinpath(NNHelferlein.DATA_DIR, path_name...)

    # download only if not already here:
    #
    # println(pname)
    # println("only: $fonly")
    # println(fname)
    if !isfile(fname) || force
        mkpath(pname)
        println("Downloading example data: $fonly")
        download(url, fname)
    end
end


"""
    function download_pretrained(h5name, force=false)

Download a hdf5 data file from github large file storage with a pretrained model.
The h5-file is saved in the `/data/pretrained/` directory of NNHelferlein.

The file is only downloaded, if not already there 
(unless `force=true`).

### Examples:
```julia
julia> download_pretrained("vgg16.h5")
```
"""
function download_pretrained(h5name, force=false)

    download_example_data("pretrained", h5name, force=force)
end
