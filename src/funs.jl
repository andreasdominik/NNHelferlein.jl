# Helper funs
#
# (c) A. Dominik, 2020

# relu(), sigm(), used from Knet!

leaky_sigm(x; l=0.01) = Knet.sigm(x) + l*x
leaky_tanh(x; l=0.01) = tanh(x) + l*x
leaky_relu(x; l=0.01) = Knet.relu(x) + l*x
