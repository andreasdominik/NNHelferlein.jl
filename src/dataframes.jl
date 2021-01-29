# Data loader and preprocessing funs for dataframes:
#
# (c) A. Dominik, 2020

"""
    dataframe_read(fname)

Read a data table from an CSV-file with one sample
per row and return a DataFrame with the data.
(ODS-support is removed because of frequent PyCall compatibility issues
of the OdsIO package).
"""
function dataframe_read(fName)

    if occursin(r".*\.ods$", fname)
        println("Reading ODS-files is no longer supported!")
        return nothing
        # return readODS(fname)
    elseif occursin(r".*\.csv$", fname)
        return readCSV(fname)
    else
        println("Error: unknown file format!")
        println("Please support csv or ods file")
        return nothing
    end
end

# function readODS(fname)
#
#     printf("Reading data from ODS: $fname")
#     return OdsIO.ods_read(fname, retType="DataFrame")
# end

function readCSV(fname)

    printf("Reading data from CSV: $fname")
    return DataFrames.DataFrame(CSV.File(fname, header=true))
end



"""
    dataframe_minibatches(data::DataFrames.DataFrame; size=256, ignore=[], teaching=:y, regression=true)

Make Knet-conform minibatches from a dataframe
with one sample per row.

+ `ignore`: defines a list of column names to be ignored
+ `teaching`: defines the column with teaching input. Default is ":y"
+ `regression`: if `true`, the teaching input is interpreted as
                scalar value (converted to Float32); otherwise
                teaching input is used as class labels (converted
                to UInt8).
"""
function dataframe_minibatches(data; size=256, ignore=[], teaching=:y, regression=true)

    if teaching == nothing
        x = Matrix(transpose(Array{Float32}(select(data, Not(ignore)))))
        return Knet.minibatch(x, size)
    else
        push!(ignore, teaching)
        x = Matrix(transpose(Array{Float32}(select(data, Not(ignore)))))
        if regression
            y = Matrix(transpose(Array{Float32}(data[teaching])))
        else
            y = Matrix(transpose(Array{UInt8}(data[teaching]))) .+ UInt8(1)
        end
        return Knet.minibatch(x, y, size, partial=true)
    end
end


"""
    function dataframe_split(df::DataFrames.DataFrame; teaching=:y, fr=0.2, balanced=true)

Split data, organised row-wise in a DataFrame into train and valid sets.

### Arguments:
+ `df`: data
+ `teaching`: name or index of column with teaching input (y)
+ `fr`: fraction of data to be used for validation
+ `balanced`: if `true`, result datasets will be balanced by oversampling.
              Returned datasets will be bigger as expected
              but include the same numbers of samples for each class.
"""
function dataframe_split(df::DataFrames.DataFrame; teaching=:y,
                         fr=0.2, balanced=false)

    ((trn,ytrn), (vld,yvld)) = do_split(df, df[teaching], at=fr)

    if balanced
        (trn,ytrn) = do_balance(trn, ytrn)
        (vld,yvld) = do_balance(vld, yvld)
    end

    # classes = unique(df[teaching])
    # n_classes = length(classes)
    # classCounts = [nrow(filter(r -> r[teaching] == c, df)) for c in classes]
    #
    # validCounts = Int.(round.(classCounts .* fr))
    # balancedValid = maximum(validCounts)
    # trainCounts = classCounts .- validCounts
    # balancedTrain = maximum(trainCounts)
    #
    # train = similar(df, 0)
    # valid = similar(df, 0)
    # for i in 1:n_classes
    #     class_idx = findall(c -> c == classes[i], df[teaching])
    #     valid_idx = sample(class_idx, validCounts[i], replace=false)
    #     classValid = df[valid_idx, :]
    #     train_idx = filter(x -> !(x ∈ valid_idx), class_idx)
    #     classTrain = df[train_idx,:]
    #
    #     # expand if balanced:
    #     #
    #     if balanced
    #         n_missing = balancedValid - validCounts[i]
    #         fillup = sample(1:nrow(classValid), n_missing, replace=true)
    #         append!(classValid, classValid[fillup,:])
    #
    #         n_missing = balancedTrain - trainCounts[i]
    #         fillup = sample(1:nrow(classTrain), n_missing, replace=true)
    #         append!(classTrain, classTrain[fillup,:])
    #     end
    #
    #     append!(train, classTrain)
    #     append!(valid, classValid)
    # end
    #
    # train = train[sample(1:nrow(train), nrow(train), replace=false),:]
    # valid = valid[sample(1:nrow(valid), nrow(valid), replace=false),:]


    # println(" ")
    # println("Split dataset to training and validation data with classes:")
    # if balanced
    #     for c in classes
    #         println("$c, training: $balancedTrain, validation: $balancedValid")
    #     end
    # else
    #     for i in 1:n_classes
    #         println("$(classes[i]), training: $(trainCounts[i]), validation: $(validCounts[i])")
    #     end
    # end

    return trn, vld
end
