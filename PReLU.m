## "Capa" PReLU, 
classdef PReLU < handle
  properties    
    ## Resultados despu�s de la propagaci�n hacia adelante
    outputs=[];
    
    ## Resultados despu�s de la propagaci�n hacia atr�s
    gradient=[];
    gradientA=[];
    
    ## Alphas de la capa de activaci�n
    A=[];

  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=PReLU()
      s.outputs=[];
      s.gradient=[];
      
      s.A=[];
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
      g=s.gradientA;
    endfunction
    
    ## Retorne el estado aprendido
    function st=state(s)
      st=s.A;
    endfunction
    
    ## Reescriba el estado aprendido
    function setState(s,A)
      s.A=A;
    endfunction
    
    ## Propagaci�n hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = funct_PReLU(s.A,a);
      y=s.outputs;
      
      # limpie el gradiente en el paso hacia adelante
      s.gradient = [];
      s.gradientA = [];
    endfunction
    
    ## Propagacion hacia atr�s recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de PReLU no compatible con forward previo");
      endif
      localGrad = s.outputs.*(1-s.outputs);
      s.gradient = localGrad.*dLds;
      s.gradientA = localGrad.*dLds;

      g=s.gradient;
    endfunction

  endmethods
  
endclassdef