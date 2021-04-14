## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Capa ejemplo
##
## Este código es un ejemplo de implementación de una capa, con todos los métodos
## e interfaces que deben respetarse.
classdef layer_template < handle

  ## En GNU/Octave "< handle" indica que la clase se deriva de handle
  ## lo que evita que cada vez que se llame un método se cree un 
  ## objeto nuevo.  Es decir, en esta clase forward y backward alternan
  ## la instancia actual y no una copia, como sería el caso si no
  ## se usara "handle".

  properties
    ## Ejemplo Número de unidades (neuronas) en la capa
    ## (solo un ejemplo, puede borrarse)
    units=0;

  endproperties

  methods
    ## Constructor inicializa todo vacío
    function s=layer_template(units)
      if (nargin > 0)
        s.units=units;
      else
        s.units=0;
      endif

      ## TODO: Inicialice sus propiedades aquí
    endfunction

    ## Inicializa el estado de la capa (p.ej. los pesos si los hay)
    ##
    ## La función devuelve la dimensión de la salida de la capa y recibe
    ## la dimensión de los datos a la entrada de la capa
    function outSize=init(s,inputSize)
      outSize=inputSize;
    endfunction
   
    ## Retorna true si la capa tiene un estado que adaptar (como pesos).
    ##
    ## En ese caso, es necesario tener las funciones stateGradient(),
    ## state() y setState()
    function st=hasState(s)
      st=false;
    endfunction
   
   
    ## Si hasState() retorna false, las siguientes tres funciones 
    ## pueden borrarse:
    ## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv 
  
    ## Retorne el gradiente del estado, que existe solo si esta capa tiene
    ## algún estado que debe ser aprendido
    ##
    ## Este gradiente es utilizado por el modelo para actualizar el estado
    ## 
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.
    function g=stateGradient(s)
      g=[];
    endfunction
    
    ## Retorne el estado aprendido
    ## 
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.
    function st=state(s)
      st=[];
    endfunction
    
    ## Reescriba el estado aprendido
    ##
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.    
    function setState(s,W)
    endfunction

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 

 
    ## Propagación hacia adelante realiza W*x
    ## X puede ser un vector columna o una matriz.
    ##
    ## Si X es un vector columna es interpretado como un dato.  Si X
    ## es una matriz, se asume que es una matriz de diseño convencional,
    ## con cada dato en una fila.  
    ##
    ## El parámetro 'prediction' permite determinar si este método
    ## está siendo llamado en el proceso de entrenamiento (false) o en el
    ## proceso de predicción (true)      
    function y=forward(s,X,prediction=false)
      y=X;
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(s,dLds)      
      g=dLds;
    endfunction
  endmethods
endclassdef
