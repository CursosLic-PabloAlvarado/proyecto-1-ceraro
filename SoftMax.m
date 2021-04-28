## "Capa" SofMax, 
classdef SoftMax < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    ## Resultados después de la propagación hacia atrás
    gradient=[];
  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=SoftMax()
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
      s.outputs = funct_SoftMax(a);
      y=s.outputs;
      s.gradient = [];
    endfunction
    
    ## Propagacion hacia atrás recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de SoftMax no compatible con forward previo");
      endif
      #localGrad =  
 
      one_s=ones(columns(s.outputs),1);
      s.gradient = s.outputs.*dLds-((sum((s.outputs.*dLds)')')*one_s').*s.outputs;

      g=s.gradient;
    endfunction
    
  endmethods
  
endclassdef