## "Capa" ReLU, 
classdef ReLU < handle
  properties    
    ## Resultados despu�s de la propagaci�n hacia adelante
    outputs=[];
    ## Resultados despu�s de la propagaci�n hacia atr�s
    gradient=[];
  endproperties
endclassdef