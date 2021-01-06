function [w1,w2,y2,epoch] = pmc_ufg()
  epsilon = 0.0001;
  n = 0.5;
  epoch = 0;
  x = [0.05 0.1];
  d = [0.01 0.99];
  w1 = [0.15 0.2; 0.25 0.3];
  w2 = [0.4 0.45; 0.5 0.55];
  b1 = 0.35;
  b2 = 0.6;
  [i1, i2, y1, y2] = forward(w1,w2,b1,b2,x);

  while epoch <= 10000
    prime = sigmoidPrime(y2);
    sig2 = -(d'-y2).*sigmoidPrime(y2);
    w2 = w2 - n * sig2 * y1';
    
    sig1 = (w2'*sig2).*sigmoidPrime(y1);
    w1 = w1 - n * sig1 * x;
    [i1, i2, y1, y2] = forward(w1,w2,b1,b2,x);
    epoch = epoch + 1;
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
