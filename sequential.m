## Copyright (C) 2021 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5852 Introducción al Reconocimiento de Patrones
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Modelo secuencial
## 
## Esta clase encapsula una red neuronal hacia adelante, con métodos para
## agregar capas, almacenar, cargar, entrenar y predecir.
##
## El método add() permite agregar capas al modelo.  La primera capa debe
## ser del tipo "input_layer" para indicar la dimensión de los datos de entrada.
##
## Luego se agregan capas de normalización, combinación, activación.
## 
## La última capa de la red debe ser una capa de error o pérdida.
classdef sequential < handle

  properties
    ## Constantes:
    layers={}; # Capas

    ## Training parameters
    nEpochs=2000;
    minibatch=128;

    alpha=0.01;     ## Learning rate
    beta = 0.95     ## Momentum: 0 para no usar momentum
    beta2 = 0.99    ## Polo de filtro de cuadrados (0: no usar Adam))
    epsilon = 1e-9; ## Evite divisiones por cero en Adam
    
    method='adam';  ## "batch", "stochastic", "momentum", "rmsprop", "adam", "autoclip"
    
    ## Last output dimension of a layer while adding layers
    lastOutput = -1;
  endproperties

  methods
    function s=sequential()
      layers={}; 
    endfunction
    
    function add(s,layer) # M�todo para agregar capas 
      ## Agregue una capa al modelo secuencial
      ## La primera capa debe ser una capa del tipo "input_layer" para así
      ## indicar la dimensión esperada de cada dato de entrada
      ## La última capa debe ser una capa con la función de pérdida/error.
      ##
      ## Esa capa será ignorada en la predicción.
      if (isa(layer,"cell"))
        add(s,layer{1}); ## Llame de nuevo la función con solo una capa
        if (length(layer)>1)
          add(s,{layer{2:end}}); ## y luego con el resto
        endif
      elseif (isa(layer,"input_layer"))
        s.lastOutput=layer.units;
        printf("Input layer configured with dimension %i\n",s.lastOutput);
      else
        if (s.lastOutput>0)
          printf("Agregando capa '%s'(%i -> ",class(layer),s.lastOutput);
          s.layers = {s.layers{:},layer};
          s.lastOutput=s.layers{end}.init(s.lastOutput);
          printf("%i)\n",s.lastOutput);
        else
          error("Debe agregar primero una capa de entrada");
        endif
      endif
    endfunction
    
    function losslog=train(s,X,Y,valSetX=[],valSetY=[])
      ## Entrene el modelo
      ## X: matriz de diseño (datos de entrenamiento en filas)
      ## y: matriz de salida, cada file codificada one-hot
      ## valSetX: set de validación (opcional) (entradas en filas)
      ## valSetY: set de validación (opcional) (salidas en filas)
      ## losslog: protocolo con loss por época, para set de
      ##          entrenamiento y opcionalmente el set de validación
      
      ## Number of layers
      numLayers = length(s.layers);
      
      if (numLayers<1)
        error("No network structure configured yet.  Layers need to be added first.\n");
      endif
      
      minibatch = s.minibatch;
      if (strcmp(s.method,"batch")) # Si estoy en que el m�todo es batch 
        minibatch=rows(X); # Entonces el minibatch tiene como tama�o todos los datos
      endif
        
      
      ## Cuántos mini-batches tiene una época
      totalBatch=int32(ceil(rows(X)/minibatch));

      ## historial de pérdidas
      losslog=[];
      
      ## Para cada época
      for ep=1:s.nEpochs
        
        ## Indices permutados aleatoriamente para tomar minilotes
        ## como muestras sin reemplazo.
        idx=randperm(rows(X));
        
        loss=0;
        
        #Indices para sacar minibatch aleatorio para m�todos de optimizaci�n 
        idx_m=randperm(rows(X)); 
        subIdx_m=idx_m((numMB-1)*minibatch+1:min(rows(X),numMB*minibatch)); 
        subX_m=X(subIdx_m,:);
        V=s.layers{i}.backward(subX); # Gradiente para inicializar
        s_m = V.^2;
        
        
        ## itere sobre todos los minibatches de la época
        for numMB=1:totalBatch; 
          subIdx=idx((numMB-1)*minibatch+1:min(rows(X),numMB*minibatch)); 
          subX=X(subIdx,:); 
          subY=Y(subIdx,:); 
          
          ## Forward prop
          y=s.layers{1}.forward(subX);
          for l=2:numLayers-1
            y=s.layers{l}.forward(y);
          endfor
          loss+=s.layers{numLayers}.forward(y,subY);
          
          ## printf("    minibatch %i/%i       \r",numMB,totalBatch);
          ## fflush(stdout);
          
          ## Back prop
          g=s.layers{numLayers}.backward(1);
          for l=numLayers-1:-1:1
            g=s.layers{l}.backward(g);
          endfor
                    
          ## Update rules
          for i=1:numLayers-1 ## Excluya capa de pérdida 
            if (s.layers{i}.hasState()) # Si la capa por la que voy tiene un estado (pesos)
           
              
              switch (s.method)
              case "batch"
                s.layers{i}.setState(s.layers{i}.state() -
                                     s.alpha*s.layers{i}.stateGradient()); # Entonces su nuevo estado es: el estado que ya tiene
                                                                           # menos alfa por el gradiente
                                     
              # "batch" y "stochastic" se diferencian en qu� tama�o de minilote uso                                        
              case "stochastic"
                s.layers{i}.setState(s.layers{i}.state() -
                                     s.alpha*s.layers{i}.stateGradient()); 
              
              ## TODO: Agregar aquí los otros métodos de optimización.
              ##       Observe que va a requerir otros arreglos de celdas
              ##       para almacenar los gradientes filtrados, sus cuadrados,
              ##       etc. para los metodos a implementar
             case "momentum"
                V = beta*V + (1-beta)*s.layers{i}.stateGradient(); ## Filter the gradient
                s.layers{i}.setState(s.layers{i}.state() - s.alpha*V);
             case "rmsprop"
                s_m = beta2*s_m + (1-beta2)*(s.layers{i}.stateGradient().^2);
                gg_m = s.layers{i}.stateGradient()./(sqrt(s_m + rmspepsilon) );
                s.layers{i}.setState(s.layers{i}.state() - s.alpha*gg_m);
             
              otherwise
                error("Método de optimización desconocido: %s",method);
              endswitch
              
            endif # si tiene estado
          endfor  # update rules for all layers
          
        endfor ## for each minibatch
        
        printf("Epoch %i/%i Loss: %f\n",ep,s.nEpochs,loss);
        fflush(stdout);

        if (isempty(valSetX)) ## Si no hay datos de validación
          losslog = vertcat(losslog,[loss]); ## Solo guarde el loss de entrenamiento
        else
          [vY,vL]=computeLoss(s,valSetX,valSetY);  ## Calcule la pérdida con la validación
          losslog = vertcat(losslog,[loss vL]);    ## Guarde par train/validation
        endif
      endfor ## for each epoch
      
    endfunction # de la funci�n train
    
    
    ## Predicción con modelo preentrenado
    function y=test(s,X)
      numLayers=length(s.layers);
      
      y=s.layers{1}.forward(X,true); % true indica que es predicción
      for l=2:numLayers-1
        y=s.layers{l}.forward(y,true); % true indica que es predicción
      endfor
      
    endfunction

    ## Predicción con modelo preentrenado # Es para hacer pruebas
    function [y,loss]=computeLoss(s,vX,vY)
      numLayers=length(s.layers);
    
      ## Forward propagation
      y=s.layers{1}.forward(vX);
      for l=2:numLayers-1 # Pero dejando la capa de error por fuera porque no interesa para predecir 
        y=s.layers{l}.forward(y);
      endfor
      loss=s.layers{numLayers}.forward(y,vY);
      
    endfunction
    

    function layer=convertStructToLayer(s,structure,layertype)
      ## Método usado para coercionar la estructura s en una clase de tipo 
      ## layertype.
      ##
      ## Es necesaria para solventar el problema de que octave no puede 
      ## serializar classdef aún. 
      layer=eval(layertype);
      for fn=fieldnames(structure)'
        try
          layer.(fn{1}) = structure.(fn{1});
        catch
          warning("Could not copy field %s",fn{1});
        end_try_catch 
      endfor
    endfunction
    
    
    function save(s,file)
      ## Guarde red en el archivo.  Posteriormente puede cargar el archivo
      ## con load()
      ##
      ## Octave convierte las classdef a struct y por tanto pierde el tipo
      ## concreto de cada capa.
      ##
      ## Como camino alterno almacenamos los nombres de los tipos primero,
      ## para luego poder recrearlos, y una vez que se tienen instancias
      ## vacías podemos convertir las estructuras almacenadas en las clases
      ## concretas.
      
      ## Extraemos primero los nombres de las clases en un cell-array
      ## y convertimos las capas a estructuras de octave
      names={};
      layers={};
      warning('off','Octave:classdef-to-struct');

      for i=1:length(s.layers)
        names = { names{:} , class(s.layers{i}) };
        layers = { layers{:}, struct(s.layers{i}) };
      endfor

      ## save no entiende atributos de una clase, así que necesitamos
      ## pasar los parámetros de la clase a una estructura
      param.nEpochs=s.nEpochs;
      param.minibatch=s.minibatch;
      param.alpha=s.alpha;
      param.beta=s.beta;
      param.beta2=s.beta2;
      param.epsilon=s.epsilon;
      param.method=s.method;  
      
      save("-v7",file,"param","names","layers");      
    endfunction

    function o=load(s,file)
           
      ## Cargue red desde el archivo almacenado con save.
      names={};
      layers={};
      param=[];
      
      load("-v7",file,"param","names","layers");
            
      if (length(names) != length(layers))
        error("Corrupted file.  Inconsistent number of stored layers and types");
        return
      endif

      for fn=fieldnames(param)'
        try
          s.(fn{1}) = param.(fn{1});
        catch
          warning("Could not copy field %s",fn{1});
        end_try_catch 
      endfor
      
      ## De los nombres, recreemos las instancias con los tipos correctos      
      for i=1:length(names)
        printf("Loading layer %s\n",names{i});
        s.layers{i}=s.convertStructToLayer(layers{i},names{i});
      endfor
    endfunction

  endmethods
endclassdef
