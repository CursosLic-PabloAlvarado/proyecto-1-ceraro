## "Capa" PReLU, 
classdef PReLU < handle
  properties    
    ## Resultados despu�s de la propagaci�n hacia adelante
    outputs=[];
    ## Resultados despu�s de la propagaci�n hacia atr�s
    gradient=[];
    gradientA=[];
  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=PReLU()
      s.outputs=[];
      s.gradient=[];
      s.gradientA=[];
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
    
    ## Propagaci�n hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = funct_PReLU(a);
      y=s.outputs;
      s.gradient = [];
    endfunction
    
    ## Propagacion hacia atr�s recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de PReLU no compatible con forward previo");
      endif
      localGrad = s.outputs.*(1-s.outputs);
      s.gradient = localGrad.*dLds;

      g=s.gradient;
    endfunction

  endmethods
  
endclassdef