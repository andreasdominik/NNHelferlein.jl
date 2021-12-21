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
```
"""
function init0(siz...)

    x = zeros(Float32, siz...)

    if CUDA.functional()
        return Knet.KnetArray(x)
    else
        return x
    end
end




"""
    function convert2KnetArray(x)

Convert an array `x` to a `KnetArray{Float32}` or `KnetArray{Int32}`` 
only in GPU context
(if `CUDA.functional()`) or to an `Array{Float32}` / `Array{Int32}` otherwise.
"""
function convert2KnetArray(x)

    # check if GPU and accept all type of Array-like x:
    #
#    if eltype(x) <: Int
#        out_type = Int32
#    else 
#        out_type = Float32
#    end

    if CUDA.functional()
        return Knet.KnetArray{Float32}(Array(x))
    else
        return Array{Float32}(x)
    end
end



"""
function blowup_array(x, n)

Blow up an array `x` with an additional dimension
and repeat the content of the array `n` times.

### Arguments:
+ `x`: Array of any dimension
+ `n`: number of repeats. ´n == 1´ will return an
array with an addritional dimension of size 1.


### Examples:

```Julia
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
    out = x
    i = 1
    while i < n
        out = cat1d(out,x)
        i += 1
    end
    return reshape(out, siz...,:)
end


"""
function recycle_array(x, n)

Recycle an array `x` along the last dimension
and repeat the content of the array `n` times.
The number of dims stays unchanged, but the array
valueas are repeated `n` times.

### Arguments:
+ `x`: Array of any dimension
+ `n`: number of repeats. ´n == 1´ will return an unchanged
        array.

### Examples:

```Julia
julia> recycle_array([1,2],3)
6-element Array{Int64,1}:
 1
 2
 1
 2
 1
 2

julia> x = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> recycle_array(x,3)
2×6 Array{Int64,2}:
 1  2  1  2  1  2
 3  4  3  4  3  4
```
"""
function recycle_array(x, n)

    dim = ndims(x)
    out = x
    i = 1
    while i < n
        out = cat(out,x, dims=dim)
        i += 1
    end
    return out
end



# split the iterator in 2 parts at at
# and return the first and the second part as separate iterators.
#
# function split_iterator(itr, at)
#
#     len = length(itr)
#     last_trn = Int(round(len * at))
#     return Iterators.take(itr, last_trn), Iterators.drop(itr, last_trn)
# end


"""
    function de_embed(x; remove_dim=true)

Replace the maximum of the first dimension of an n-dimensional array
by its index (aka argmax()).
If `remove_dim=false` the first dim is preserved with size=1; otherwise
the returned array has the first dimension removed.
"""
function de_embed(x; remove_dim=true)

    siz = size(x)
    depth = siz[1]
    siz = siz[2:end]

    x = reshape(x, depth, :)
    x = softmax(x, dims=1)
    x = [argmax(x[:,i]) for i in 1:size(x)[2]]
    if remove_dim
        return reshape(x, siz...)
    else
        return reshape(x, 1,siz...)
    end
end
