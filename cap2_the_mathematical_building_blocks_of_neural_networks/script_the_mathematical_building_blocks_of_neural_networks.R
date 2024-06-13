########################################################################
### Capítulo 2 - The Mathematical Building Blocks of Neural Networks ###
########################################################################

#Instalando e carregando as bibliotecas necessárias
install.packages("keras") 
install.packages("tensorflow")
install.packages("reticulate")

library(keras)
library(tensorflow)
library(reticulate)

#Criando o ambiente virtual para instalação do python e keras

virtualenv_create("r-reticulate", python = install_python())
install_keras(envname = "r-reticulate", version = "2.16.1")

#Carregando a base MNIST localmente

#Não consegui acessar o link diretamente devido ao protocolo de certificação SSL
#Por este motivo baiexei o arquivo mnist.npz e salvando na pasta do livro e fiz a abertura diretamente
caminho <- setwd("/Users/lincoln/Documents/GitHub/livro_deep_learning_with_R_2edition/cap2_the_mathematical_building_blocks_of_neural_networks")
mnist <- dataset_mnist(path = paste(caminho,"/mnist.npz", sep=""))

#Analisando a base MNIST
mnist

digit <- mnist$train$x[1,,]
plot(as.raster(digit, max = 255))







use_condaenv("tf-m1", required = TRUE)
use_condaenv("r-reticulate", required = TRUE)


use_condaenv("tf-m1", required = TRUE)
