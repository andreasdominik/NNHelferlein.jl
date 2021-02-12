# balance a dataset by oversampling:
# works for vectors, arrays (by last dim) and dataframes (by row).
# returns new x, y-Tuple:
#
function do_balance(x,y)

    return MLDataUtils.oversample((x, y))
end


function do_split(x,y; at=0.5)

    # return MLDataUtils.splitobs(MLDataUtils.shuffleobs((x, y)), at=at)
    return MLDataUtils.stratifiedobs((x, y), p=at)
end

function do_shuffle(x,y)

    return collect(MLDataUtils.shuffleobs((x, y)))
end


"""
    function crop_array(x, crop_sizes)

Crop a n-dimensional array to the given size. Cropping is always
centered (i.e. a margin is removed).

### Arguments:
+ `x`: n-dim AbstractArray
+ `crop_sizes`: Tuple of target sizes to which the array is cropped.
        Allowed values are Int or `:`. If `crop_sizes` defines less
        dims as x has, the remaining dims will not be cropped (assuming `:`).
        If a demanded crop size is bigger as the actual size of x,
        it is ignored.
"""
function crop_array(x, crop_size)

    x_size = size(x)
    ranges = []

    for (cs,s) in zip(crop_size, x_size)
        if cs isa Colon
            push!(ranges, :)
        elseif cs >= s
            push!(ranges, :)
        else
            margin = fld(s-cs, 2)
            push!(ranges, margin+1:margin + cs)
        end
    end
    while length(ranges) < length(x_size)
        push!(ranges, :)
    end
    return x[ranges...]
end




"""
    function init0(siz...)

Initialise a vector or array of size `siz` with zeros.
If a GPU is detected type of the returned value is `KnetArray{Float32}`,
otherwise `Array{Float32}`.

### Examples:
```
julia> init0(2,10)
2×10 Array{Float32,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

 julia> init0(0,10)
 0×10 Array{Float32,2}

 julia> init0((2,10))
2×10 Array{Float32,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function init0(siz...)

    x = zeros(Float32, siz...)

    if CUDA.functional()
        return KnetArray(x)
    else
        return x
    end
end


"""
function blowup_array(x, n)

Blow up an array `x` with an additional dimension
and repeat the content of the array `n` times.

### Examples:
```
julia> x = [1,2,3,4]; blowup_array(x, 3)
4×3 Array{Int64,2}:
 1  1  1
 2  2  2
 3  3  3
 4  4  4

julia> x = [1 2; 3 4]; blowup_array(x, 3)
2×2×3 Array{Int64,3}:
[:, :, 1] =
 1  2
 3  4

[:, :, 2] =
 1  2
 3  4

[:, :, 3] =
 1  2
 3  4
 ```
"""
function blowup_array(x, n)

    siz = size(x)
    return reshape(repeat(vec(x), n), siz...,:)
end
