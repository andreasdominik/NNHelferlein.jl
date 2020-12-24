# balance a dataset by oversampling:
# works for vectors, arrays (by last dim) and dataframes (by row).
# returns new x, y-Tuple:
#
function d_balance(x,y)

    return MLDataUtils.oversample((x, y))
end


function d_split(x,y, at=0.5)

    return MLDataUtils.splitobs(shuffleobs((x, y)), at=at)
end

function d_shuffle(x,y)

    return MLDataUtils.shuffleobs((x, y))
end
