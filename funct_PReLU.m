function l=funct_PReLU(alpha_PReLU,x)
  l=max(alpha_PReLU.*x,x); # alpha_PReLU es un par�metro aprendido con un valor entre 0 y 1  
endfunction