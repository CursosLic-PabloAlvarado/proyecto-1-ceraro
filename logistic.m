## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Función logística 1/1+exp(-x)
function l=logistic(x)
  l=1./(1+exp(-x));
endfunction
