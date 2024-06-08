
install.packages("keras") 

library(reticulate)
virtualenv_create("r-reticulate", python = install_python())

library(keras)
install_keras(envname = "r-reticulate", version = "2.16.1")

mnist <- dataset_mnist()

setwd("/Users/lincoln/Documents/GitHub/livro_deep_learning_with_R_2edition/cap2_the_mathematical_building_blocks_of_neural_networks")

