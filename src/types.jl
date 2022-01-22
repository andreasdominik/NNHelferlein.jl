
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


"""
    struct PartialIterator <: DataLoader

The PartialIterator wraps any iterator and will only iterate the states
specified in the list `indices`. 

### Constuctors

    PartialIterator(inner, indices; shuffle=true) 

Type of the states must match
the states of the wrapped iterator `inner`. A `nothing` element may be 
given to specify the first iterator element.

If `shuffle==true`, the list of indices are shuffled every time the
PartialIterator is started.
"""
mutable struct PartialIterator <: DataLoader
    inner
    indices
    l
    shuffle
    PartialIterator(inner, indices; shuffle=true) = new(inner, indices, length(indices), shuffle)
end

function Base.iterate(it::PartialIterator, state=0)
    
    if it.shuffle && state == 0
        it.indices = shuffle(it.indices)
    end
    
    if state >= it.l
        return nothing
    else
        state += 1
        inner_state = it.indices[state]
        
        if inner_state == nothing
            return iterate(it.inner,)[1], state
        else
            return iterate(it.inner, inner_state)[1], state
        end
    end
end

Base.length(it::PartialIterator) = it.l
Base.eltype(it::PartialIterator) = eltype(first(it.inner))