function val=funct_SoftMax(x)
  val = exp(x-max(x).*ones(rows(x),1));
  nor = sum(val);
  val = val ./ nor;
endfunction