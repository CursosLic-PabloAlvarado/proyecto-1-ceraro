function l=funct_PReLU(x)
  l=max(alpha_PReLU*x,x); # alpha_PReLU es un parámetro aprendido con un valor entre 0 y 1  
endfunction