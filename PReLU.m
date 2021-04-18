## "Capa" PReLU, 
classdef PReLU < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[];
    
    ## Resultados después de la propagación hacia atrás
    gradient=[];
    gradientA=[];
    
    ## Alphas de la capa de activación
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

    ## En funciones de activación el init no hace mayor cosa más que
    ## indicar que la dimensión de la salida es la misma que la entrada.
    ##
    ## La función devuelve la dimensión de la salida de la capa
    function outSize=init(s,inputSize)
      outSize=inputSize;
    endfunction 

    ## Retorna false si la capa no tiene un estado que adaptar
    function st=hasState(s)
      st=true;
    endfunction
    
    ## Retorne el gradiente del estado, que existe solo si esta capa tiene
    ## algÃºn estado que debe ser aprendido
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
    
    ## Propagación hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = funct_PReLU(s.A,a);
      y=s.outputs;
      
      # limpie el gradiente en el paso hacia adelante
      s.gradient = [];
      s.gradientA = [];
    endfunction
    
    ## Propagacion hacia atrás recibe dL/ds de siguientes nodos
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