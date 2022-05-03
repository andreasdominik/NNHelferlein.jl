# Helper funs
#
# (c) A. Dominik, 2020

# relu(), sigm(), used from Knet!

leaky_sigm(x; l=0.01) = Knet.sigm(x) .+ eltype(value(x))(l) .* x
leaky_tanh(x; l=0.01) = tanh(x) .+ eltype(value(x))(l) .* x
leaky_relu(x; l=0.01) = Knet.relu(x) .+ eltype(value(x))(l) .* x
