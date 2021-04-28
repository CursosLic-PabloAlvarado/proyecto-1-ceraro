## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## "Capa" sigmoide, que aplica la función logística
classdef sigmoide < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    ## Resultados después de la propagación hacia atrás
    gradient=[];
  endproperties

  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=sigmoide()# Se inicializan las propiedades de la clase sin nada
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## En funciones de activación el init no hace mayor cosa más que
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
        
    ## Propagación hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = logistic(a);
      y=s.outputs;
      s.gradient = []; # Le digo que el gradiente aun no lo he calculado
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos
    function g=backward(s,dLds) # dLds: gradiente del error (L) respecto a la salida (s)
      if (size(dLds)!=size(s.outputs))
        error("backward de sigmoide no compatible con forward previo");
      endif
      localGrad = s.outputs.*(1-s.outputs); # s.outputs es la salida que yo almacen� justo en la forward
      s.gradient = localGrad.*dLds; # .* porque este es el t�pico caso en el que el Jacobiano es diagonal
      
      g=s.gradient;
    endfunction
  endmethods
endclassdef
