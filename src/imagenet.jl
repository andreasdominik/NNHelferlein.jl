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
    ia = Float32.(permutedims(channelview(img), (3,2,1)) .* 255.0)

    y = zeros(Float32, size(ia))
    y[:,:,1] .= ia[:,:,3] .- r
    y[:,:,2] .= ia[:,:,2] .- g
    y[:,:,3] .= ia[:,:,1] .- b
    return y
end
