function val=funct_SoftMax(x)
  val = exp(x);
  nor = sum(val);
  val = val ./ nor;
endfunction