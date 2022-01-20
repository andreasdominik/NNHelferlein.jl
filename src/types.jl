
"""
    abstract type DataLoader

Mother type for minibatch iterators.
"""
abstract type DataLoader end

"""
    struct SequenceData <: DataLoader

Type for a generic minibatch iterator.
All NNHelferlein models accept minibatches of type `DataLoader`.

### Constructors:

    SequenceData(x; shuffle=true)

+ `x`: List, Array or other iterable object with the minibatches
+ `shuffle`: if `true`, minibatches are shuffled every epoch.
"""
mutable struct SequenceData <: DataLoader
    mbs
    l
    indices
    shuffle
    SequenceData(x; shuffle=true) = new(x, length(x), collect(1:length(x)), shuffle)
end

function Base.iterate(it::SequenceData, state=0)

    # shuffle if first call:
    #
    if it.shuffle && state == 0
        it.indices = Random.randperm(it.l)
    end
        
    if state >= it.l
        return nothing
    else
        state += 1
        return it.mbs[it.indices[state]], state
    end
end

Base.length(it::SequenceData) = it.l
Base.eltype(it::SequenceData) = eltype(first(it.mbs))

