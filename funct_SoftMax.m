function val=funct_SoftMax(x)
  if columns(x)==1
    val = funct_SoftMax(x'); # Es para cuando x llega como vector
  else
    val = exp(x-max(x,[],2)); # Es para una matriz de dise�o
                              # Octave aplica un broadcasting por eso no es necesario multiplicaciones por vectores de 1's
    nor = sum(val);
    val = val ./ nor;
  endif
endfunction