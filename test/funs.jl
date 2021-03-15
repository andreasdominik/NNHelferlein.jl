
function test_leaky()
    return leaky_sigm(1000) > 1.0 &&
           leaky_relu(-10) < 0.0 &&
           leaky_tanh(1000) > 1.0
end
