## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducci�n al Reconocimiento de Patrones
## Escuela de Ingenier�a Electr�nica
## Tecnol�gico de Costa Rica

## Capa densa sin sesgo
##
## Este c�digo es un ejemplo de implementaci�n de una capa.
## 
## La capa implementada es totalmente conectada, pero sin sesgo.
## Es decir, se calcula un plano de separaci�n que siempre pasa por
## el origen.
##
## En principio, X es una matriz de dise�o que tiene a los datos
## en sus FILAS.  Esta es la convenci�n.  Sin embargo, W lo que hace 
## es multiplicar a un vector (columna) x para producir al vector 
## columna y con la salida (y=Wx).  
##
## Queremos mantener la l�gica de que tanto X como Y contendr�n a los
## vectores de entrada y salida en sus filas, por lo que 
##   Y' = W X'   =>   Y = X W'
##
## Esto est� implementado para que X y Y mantengan sus naturalezas
## de matrices de dise�o, y que W tenga los pesos tal que y=Wx.  
##
## Si el m�todo forward recibe un solo vector (columna) x, el c�lculo
## se ajusta adecuadamente para producir y.  De otro modo, se asume que
## la entrada es una matriz convencional de dise�o.
classdef dense_bias < handle

  ## En GNU/Octave "< handle" indica que la clase se deriva de handle
  ## lo que evita que cada vez que se llame un m�todo se cree un 
  ## objeto nuevo.  Es decir, en esta clase forward y backward alternan
  ## la instancia actual y no una copia, como ser�a el caso si no
  ## se usara "handle".

  properties
    ## N�mero de unidades (neuronas) en la capa
    units=0;
    
    ## Pesos de la capa densa sin senso
    W=[];
    
    ## Entrada de valores en la propagaci�n hacia adelante
    inputsX=[];
    
    ## Resultados despu�s de la propagaci�n hacia atr�s
    gradientW=[];
    gradientX=[];
  endproperties

  methods
    ## Constructor inicializa todo vac�o
    function s=dense_bias(units)
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
    ## de la capa tendr� un vector de entrada con el n�mero dado de 
    ## dimensiones.
    ##
    ## La funci�n devuelve la dimensi�n de la salida de la capa
    function outSize=init(s,inputSize)
      
      cols = inputSize+1;
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
    ## alg�n estado que debe ser aprendido
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
   
    ## Propagaci�n hacia adelante realiza W*x
    function y=forward(s,X,prediction=false)
      ## X puede ser un vector columna o una matriz.
      ##
      ## Si X es un vector columna es interpretado como un dato.  Si X
      ## es una matriz, se asume que es una matriz de dise�o convencional,
      ## con cada dato en una fila.  
      ##
      ## El par�metro 'prediction' permite determinar si este m�todo
      ## est� siendo llamado en el proceso de entrenamiento (false) o en el
      ## proceso de predicci�n (true)      
      
      #s.inputsX=X;
      
      if (columns(X)==1) 
        s.inputsX=[1;X];
        y = s.W.*s.inputsX; %% Si es vector, asuma columna
      else
        s.inputsX=[ones(rows(X),1) X];
        y = s.inputsX*s.W'; %% Si es matriz de dise�o, asuma datos en filas
      endif
      
      # limpie el gradiente en el paso hacia adelante
      s.gradientX = [];
      s.gradientW = [];
    endfunction

    ## Propagaci�n hacia atr�s recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagaci�n. que ser�
    ## pasado a nodos anteriores en el grafo.
    function g=backward(s,dLds)      

      if (columns(dLds)==1)
        s.gradientW = dLds*s.inputsX';
        s.gradientX = s.W'*dLds;
        s.gradientX=s.gradientX(2:end,:);
      else
        s.gradientW = dLds'*s.inputsX;
        s.gradientX = dLds*s.W;
        s.gradientX=s.gradientX(:,2:end);
      endif
      
      g=s.gradientX;
    endfunction
  endmethods
endclassdef
