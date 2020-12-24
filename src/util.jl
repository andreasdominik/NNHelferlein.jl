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
