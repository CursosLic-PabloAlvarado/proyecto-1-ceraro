## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Capa densa sin sesgo
##
## Este código es un ejemplo de implementación de una capa.
## 
## La capa implementada es totalmente conectada, pero sin sesgo.
## Es decir, se calcula un plano de separación que siempre pasa por
## el origen.
##
## En principio, X es una matriz de diseño que tiene a los datos
## en sus FILAS.  Esta es la convención.  Sin embargo, W lo que hace 
## es multiplicar a un vector (columna) x para producir al vector 
## columna y con la salida (y=Wx).  
##
## Queremos mantener la lógica de que tanto X como Y contendrán a los
## vectores de entrada y salida en sus filas, por lo que 
##   Y' = W X'   =>   Y = X W'
##
## Esto está implementado para que X y Y mantengan sus naturalezas
## de matrices de diseño, y que W tenga los pesos tal que y=Wx.  
##
## Si el método forward recibe un solo vector (columna) x, el cálculo
## se ajusta adecuadamente para producir y.  De otro modo, se asume que
## la entrada es una matriz convencional de diseño.
classdef dense_unbiased < handle

  ## En GNU/Octave "< handle" indica que la clase se deriva de handle
  ## lo que evita que cada vez que se llame un método se cree un 
  ## objeto nuevo.  Es decir, en esta clase forward y backward alternan
  ## la instancia actual y no una copia, como sería el caso si no
  ## se usara "handle".

  properties
    ## Número de unidades (neuronas) en la capa
    units=0;
    
    ## Pesos de la capa densa sin senso
    W=[];
    
    ## Entrada de valores en la propagación hacia adelante
    inputsX=[];
    
    ## Resultados después de la propagación hacia atrás
    gradientW=[];
    gradientX=[];
  endproperties

  methods
    ## Constructor inicializa todo vacío
    function s=dense_unbiased(units)
      if (nargin > 0)
        s.units=units;
      else
        s.units=0;
      endif

      s.inputsX=[];
      s.W=[];
      
      s.gradientX=[];
      s.gradientW=[];
    endfunction

    ## Inicializa los pesos de la capa, considerando que la entrada
    ## de la capa tendrá un vector de entrada con el número dado de 
    ## dimensiones.
    ##
    ## La función devuelve la dimensión de la salida de la capa
    function outSize=init(s,inputSize)
      
      cols = inputSize;
      rows = s.units; # Cantidad de neuronas
      
      ## LeCun Normal (para selu)
      s.W=normrnd(0,1/sqrt(cols),rows,cols);
      outSize=s.units;
    endfunction
   
    ## Retorna true si la capa tiene un estado que adaptar.
    ##
    ## En ese caso, es necesario tener las funciones stateGradient(),
    ## state() y setState()
    function st=hasState(s)
      st=true;
    endfunction
   
    ## Retorne el gradiente del estado, que existe solo si esta capa tiene
    ## algún estado que debe ser aprendido
    ##
    ## Este gradiente es utilizado por el modelo para actualizar el estado
    ## 
    function g=stateGradient(s)
      g=s.gradientW;
    endfunction
    
    ## Retorne el estado aprendido
    function st=state(s)
      st=s.W;
    endfunction
    
    ## Reescriba el estado aprendido
    function setState(s,W)
      s.W=W;
    endfunction
   
    ## Propagación hacia adelante realiza W*x
    function y=forward(s,X,prediction=true)
      ## X puede ser un vector columna o una matriz.
      ##
      ## Si X es un vector columna es interpretado como un dato.  Si X
      ## es una matriz, se asume que es una matriz de diseño convencional,
      ## con cada dato en una fila.  
      ##
      ## El parámetro 'prediction' permite determinar si este método
      ## está siendo llamado en el proceso de entrenamiento (false) o en el
      ## proceso de predicción (true)      
      s.inputsX=X;
      if (columns(X)==1) 
        y = s.W*X; %% Si es vector, asuma columna
      else
        y = X*s.W'; %% Si es matriz de diseño, asuma datos en filas
      endif
      
      # limpie el gradiente en el paso hacia adelante
      s.gradientX = [];
      s.gradientW = [];
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(s,dLds)      

      if (columns(dLds)==1)
        s.gradientW = dLds*s.inputsX';
        s.gradientX = s.W'*dLds;
      else
        s.gradientW = dLds'*s.inputsX;
        s.gradientX = dLds*s.W;
      endif
      
      g=s.gradientX;
    endfunction
  endmethods
endclassdef
