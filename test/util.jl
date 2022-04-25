function test_crop_array()
    x = rand(16,16,32,8)
    y = crop_array(x,(:,:,30))

    return size(y) == (16,16,30,8)
end


function test_init0()
    x = init0(16,8,4)

    return size(x) == (16,8,4) && sum(x) ≈ 0
end

function test_convertKA()
    x = rand(8,8)
    x = convert2KnetArray(x)

    if CUDA.functional()
        return x isa Knet.KnetArray{Float32} || x isa CuArray
    else
        return x isa Array{Float32}
    end
end

function test_blowup()
    x = rand(4,4)
    y = blowup_array(x, 3)
    return size(y) == (4,4,3)
end

function test_recycle()
    x = rand(4,4)
    y = recycle_array(x, 4)
    return size(y) == (4,16)
end

function test_de_embed()
    x = rand(8,4,12)
    y = de_embed(x)
    return size(y) == (1,4,12)
end

function test_confusion_matrix()
    test_net = load(joinpath(DATA_DIR, "testdata", "dummy_mlp.jld2"))
    model = test_net["mlp"]
    data = test_net["mb"]
    human_readable = false
    return isapprox(confusion_matrix(model, data, human_readable ), [1.0 0.0; 0.454545 0.0]; atol = 0.01 )
end  

