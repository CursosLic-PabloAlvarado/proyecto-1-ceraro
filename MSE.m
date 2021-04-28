## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## "Capa" para calcular la pérdida con "ordinary least squares"
##
## Suponemos que cada fila de Y tiene un dato, para el que
## se tiene como 'ground-truth' las etiquetas Ygt.
##
## Esta capa calcula entonces la pérdida como la mitad de la suma de los
## cuadrados de las diferencias
classdef olsloss < handle
  properties
    ## Entrada en la propagación hacia adelante
    diff=[];
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    ## Resultados después de la propagación hacia atrás
    gradient=[];
  endproperties

  methods
    ## Constructor solo incializa los datos
    function s=olsloss()
      s.diff=[];
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## En funciones de perdida el init no hace mayor cosa más que
    ## indicar que la dimensión de la salida es la misma que la entrada.
    ##
    ## La función devuelve la dimensión de la salida de la capa
    function outSize=init(s,inputSize)
      outSize=inputSize;
    endfunction

    ## Retorna false si la capa no tiene un estado que adaptar
    function st=hasState(s)
      st=false;
    endfunction
    
    ## Propagación hacia adelante.
    ## 
    ## En las capas de error, se requieren dos argumentos.
    ## 
    ## Primero la salida de la última capa de la red y luego las etiquetas
    ## contra las que se comparará y se calculará la pérdida.
    ##
    ## Note que todas las otras capas solo requieren la salida de la capa anterior.
    function J=forward(s,Y,Ygt)
      if (isscalar(Ygt) && isboolean(Ygt))
        error("Capas de pérdida deben ser las últimas del grafo");
      elseif (isreal(Y) && ismatrix(Y) && (size(Y)==size(Ygt)))
        m=rows(Y)
        s.diff=Y-Ygt;
        s.outputs = (1/m)*(norm(s.diff,"fro")^2); # Frobenius norm
        J=s.outputs;
        s.gradient = [];
      else
        error("olsloss espera dos matrices reales del mismo tamaño");
      endif
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de olsloss no compatible con forward previo");
      endif
      ## Asumiendo que dLds es escalar (la salida debería serlo)
      s.gradient = (2/m)*s.diff*dLds;
      
      g=s.gradient;
    endfunction
  endmethods
endclassdef
