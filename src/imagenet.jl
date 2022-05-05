
"""
    function preproc_imagenet(img)

Image preprocessing for pre-trained ImageNet examples.
Preprocessing includes
+ bring RGB colour values into a range 0-255
+ standardise of colour values by substracting mean colour values
    (103.939, 116.779, 123.68) from RGB
+ changing colour channel sequence from RGB to BGR

Resize is **not** done, because this may be part of the
augmentation pipeline.

### Examples:
The function can be used with the image loader; for prediction
with a trained model as:
```julia
pipl = CropRatio(ratio=1.0) |> Resize(224,224)
images = mk_image_minibatch("./example_pics", 16;
                    shuffle=false, train=false,
                    aug_pipl=pipl,
                    pre_proc=preproc_imagenet)
```

And for training something like:
```julia
pipl = Either(1=>FlipX(), 1=>FlipY(), 2=>NoOp()) |>
       Rotate(-5:5) |>
       ShearX(-5:5) * ShearY(-5:5) |>
       RCropSize(224,224)

dtrn, dvld = mk_image_minibatch("./example_pics", 16;
                    split=true, fr=0.2, balanced=false,
                    shuffle=true, train=true,
                    aug_pipl=pipl,
                    pre_proc=preproc_imagenet)
```
"""
function preproc_imagenet(img)

    (r, g, b) = (103.939, 116.779, 123.68)
    img = img .* 255.0

    y = zeros(Float32, size(img))
    y[:,:,1] .= img[:,:,3] .- r
    y[:,:,2] .= img[:,:,2] .- g
    y[:,:,3] .= img[:,:,1] .- b
    return y
end


"""
    function get_imagenet_classes()

Return a list of all 1000 ImageNet class labels.
"""
function get_imagenet_classes()

    IMAGENET_CLASSES = joinpath(NNHelferlein.DATA_DIR, "imagenet", "classes.txt")

    if isfile(IMAGENET_CLASSES)
        classes = readlines(IMAGENET_CLASSES)
    else
        println("File with ImageNet class labels not found at")
        println("$IMAGENET_CLASSES.")

        classes = repeat(["?"], 1000)
    end
    return classes
end




"""
    function predict_imagenet(mdl; data, top_n=5)

Predict the ImageNet-class of images from the
predefined list of class labels.
"""
function predict_imagenet(mdl; data, top_n=5)

    classes = get_imagenet_classes()
    return predict_top5(mdl; data=data; top_n=top_n, classes=classes)
end
