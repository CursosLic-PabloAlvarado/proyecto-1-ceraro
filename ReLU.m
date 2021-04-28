## "Capa" ReLU, 
classdef ReLU < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    inputsX=[];
    ## Resultados después de la propagación hacia atrás
    gradient=[];
  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=ReLU()
      s.outputs=[];
      s.inputsX=[];
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
      s.outputs = funct_ReLU(a);
      y=s.outputs;
      s.inputsX=a;
      s.gradient = [];
    endfunction
    
    ## Propagacion hacia atrás recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de ReLU no compatible con forward previo");
      endif
      Ux=s.inputsX>=0;
      #localGrad = Ux;
      s.gradient = Ux.*dLds;
      
      g=s.gradient;
    endfunction
    
  endmethods
endclassdef
