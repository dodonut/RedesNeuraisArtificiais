function [w1,w2,y2,epoch] = pmc_ufg(x,d,w1,w2,b1,b2)
  epsilon = 0.0001;
  n = 0.5;
  epoch = 0;
  [i1, i2, y1, y2] = forward(w1,w2,b1,b2,x);
  while erro(d,y2) > epsilon
    prime = sigmoidPrime(y2);
    sig2 = -(d'-y2).*sigmoidPrime(y2);
    w2 = w2 - n * sig2 * y1';
    
    sig1 = (w2'*sig2).*sigmoidPrime(y1);
    w1 = w1 - n * sig1 * x;
    [i1, i2, y1, y2] = forward(w1,w2,b1,b2,x);
    epoch = epoch + 1;
    if mod(epoch, 500) == 0
      epoch
    endif
  endwhile
endfunction

function [x] = erro(d,y)
  x = sum(power(d'-y,2)/2);
endfunction

function [i1,i2,y1,y2] = forward(w1,w2,b1,b2,x)
  i1 = w1 * x' + b1;
  y1 = sigmoid(i1);
  i2 = w2 * y1 + b2;
  y2 = sigmoid(i2);
endfunction

function [x] = sigmoid(z)
  x = 1 ./ (1 + exp(-z));
endfunction

function [x] = sigmoidPrime(z)
  x = z.*(1-z);
endfunction
