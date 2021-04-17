## "Capa" ReLU, 
classdef ReLU < handle
  properties    
    ## Resultados despu�s de la propagaci�n hacia adelante
    outputs=[];
    ## Resultados despu�s de la propagaci�n hacia atr�s
    gradient=[];
  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=SoftMax()
      s.outputs=[];
      s.gradient=[];
    endfunction

    ## En funciones de activaci�n el init no hace mayor cosa m�s que
    ## indicar que la dimensi�n de la salida es la misma que la entrada.
    ##
    ## La funci�n devuelve la dimensi�n de la salida de la capa
    function outSize=init(s,inputSize)
      outSize=inputSize;
    endfunction 

    ## Retorna false si la capa no tiene un estado que adaptar
    function st=hasState(s)
      st=false;
    endfunction
    
    ## Propagaci�n hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = logistic(a);
      y=s.outputs;
      s.gradient = [];
    endfunction

  endmethods
  
endclassdef