## "Capa" PReLU, 
classdef PReLU < handle
  properties    
    ## Resultados después de la propagación hacia adelante
    outputs=[]; # En PReLU no es necesario almacenar las salidas porque no se usan par calcular los gradientes
    inputsX=[];
    
    ## Resultados después de la propagación hacia atrás
    gradientX=[];
    gradientA=[];
    
    ## Alphas de la capa de activación
    A=[];
    Ux=[]; # Función escalón unitario 

  endproperties
  
  methods
    ## Constructor ejecuta un forward si se le pasan datos
    function s=PReLU()
      s.outputs=[];
      s.inputsX=[];
      s.gradientX=[];
      
      s.A=[];
      s.Ux=[];
      s.gradientA=[];
    endfunction

    ## En funciones de activación el init no hace mayor cosa más que
    ## indicar que la dimensión de la salida es la misma que la entrada.
    ##
    ## La función devuelve la dimensión de la salida de la capa
    function outSize=init(s,inputSize)
      outSize=inputSize;
      
      s.A = 0; # Así se comporta como ReLU
               # Se comporta como Leaky ReLU con s.A = 0.1 
               
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
       
      s.A=min(max(A,0),.99); # min y max para evitar alphas negativos y alphas igual a 1
                             # Funciona para cuando las alphas sean dadas como vector pero también como escalar 
    endfunction
    
    ## Propagación hacia adelante
    function y=forward(s,a,prediction=false)
      s.outputs = funct_PReLU(s.A,a);
      y=s.outputs;
      
      s.inputsX = a; #  Almaceno las entradas
      s.Ux = a>=0;
      
      # limpie el gradiente en el paso hacia adelante
      s.gradientX = [];
      s.gradientA = [];
    endfunction
    
    ## Propagacion hacia atrás recibe dL/ds de siguientes nodos
    function g=backward(s,dLds)
      if (size(dLds)!=size(s.outputs))
        error("backward de PReLU no compatible con forward previo");
      endif
      localGradX= s.A + (1-s.A).*s.Ux; # Revisado a mano: Si alpha que llega es valor escalar entonces está correcto
                                      # No hace falta el .* porque aunque s.A es escalar, octave hace un broadcasting
                                      # De la 2° solución obtenida s.A sería un vector por lo que sí necesito el .*   
      localGradA= (1-s.Ux).*s.inputsX; 
      # Dado que la función de PReLU consiste en sacar un máximo entre dos valores (compuerta max)
      s.gradientA = sum(localGradA.*dLds);
      s.gradientX = localGradX.*dLds;  #Revisado a mano: está correcto

      g=s.gradientX;
    endfunction

  endmethods
  
endclassdef