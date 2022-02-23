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
    println(pname)
    println("only: $fonly")
    println(fname)
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


function download_example_data()

    data_files = [
        ["Images64", "car.png"],
        ["Images64", "queen.png"],
        ["Images64", "spock.png"],
        ["Images64", "telephone.png"],
        ["Images64", "tree.png"],

        ["MIT-normal_sinus_rhythm", "16265.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16272.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16273.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16420.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16483.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16539.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16773.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16786.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "16795.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "17052.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "17453.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "18177.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "18184.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "19088.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "19090.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "19093.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "19140.ecg.gz"],
        ["MIT-normal_sinus_rhythm", "19830.ecg.gz"],

        ["elecat", "cat.jpg"],
        ["elecat", "elephant.jpg"],

        ["elecat_224", "cat-224.jpg"],
        ["elecat_224", "elephant-224.jpg"],

        ["flowers", "daisy", "5547758_eea9edfd54_n_sqr.png"],
        ["flowers", "daisy", "5673551_01d1ea993e_n_sqr.png"],
        ["flowers", "daisy", "5673551_01d1ea993e_n_sqr_copy.png"],
        ["flowers", "daisy", "5673551_01d1ea993e_n_sqr_copy_copy.png"],
        
        ["flowers", "rose", "22679076_bdb4c24401_m_sqr.png"],
        ["flowers", "rose", "99383371_37a5ac12a3_n_sqr.png"],

        ["flowers", "tulip", "11746080_963537acdc_sqr.png"],
        ["flowers", "tulip", ""]
    ]
end


#    https://github.com/andreasdominik/NNHelferlein.jl/raw/main/data/MIT-normal_sinus_rhythm/16265.ecg.gz