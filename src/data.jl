# Data loader and preprocessing funs:
#
# (c) A. Dominik, 2020

"""
    readDataTable(fname)

Read a data table from an ODS- or CSV-file with one sample
per row and return a DataFrame with the data.
"""
function readDataTable(fName)

    if occursin(r".*\.ods$", fname)
        return readODS(fname)
    elseif occursin(r".*\.csv$", fname)
        return readCSV(fname)
    else
        println("Error: unknown file format!")
        println("Please support csv or ods file")
        return nothing
    end
end

function readODS(fname)

    printf("Reading data from ODS: $fname")
    return OdsIO.ods_read(fname, retType="DataFrame")
end

function readCSV(fname)

    printf("Reading data from CSV: $fname")
    return DataFrame(CSV.File(fname, header=true))
end
