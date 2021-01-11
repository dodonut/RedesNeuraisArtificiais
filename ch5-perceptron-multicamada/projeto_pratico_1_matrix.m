  function [w1, w2, eqm_curr,epoch, yy2] = projeto_pratico_1_matrix(X,d)
    epsilon = 10^-6;
    min = -2.4;
    max = 2.4;
    n = 0.05;
    epoch = 0;
    eqm_bef = 0;
    eqm_curr = 1;
    amount = size(X,1);
    w1 = (min + (max-min) * rand(10,3))/amount;
    w2 = (min + (max-min) * rand(1,10))/amount;
##    w1 = rand(10,3);
##    w2 = rand(1,10);
    b1 = rand(10,amount);
    b2 = rand(1,amount);
    n_bias = 0.001;
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy2 = [];
      
      [i1,i2,y1,y2] = forward(w1,w2,b1,b2,X);
      yy2 = [yy2 y2];

      sig2 = -(d'-y2).*sigmoidPrime(y2);
      w2 = w2 - n * sig2 * y1';
      b2 = b2 - n_bias * b2 .* sig2;

      sig1 = (w2'*sig2).*sigmoidPrime(y1);
      w1 = w1 - n * sig1 * X;
      b1 = b1 - n_bias * b1 .* sig1;
      
      epoch = epoch + 1;
      eqm_curr = eqm(yy2,d,amount);
      if mod(epoch,50) == 0
        epoch
        eqm_curr
      endif
    endwhile
    
  endfunction
  
  function [x] = eqm(Y,d,amount)
    Y = Y';
    x = 0;
    for i = 1:amount
      yk = Y(i,:);
      dk = d(i,:);
      p = power(dk-yk,2)/2;
      x = x + sum(p);
    endfor
    x = x/amount;
  
  endfunction
  
  function [x] = tanhPrime(z)
    x = 1.0 - power(tanh(z),2);
  endfunction
  
  function [i1,i2,y1,y2] = forward(w1,w2,b1,b2,x)
    i1 = w1*x' + b1;
    y1 = sigmoid(i1);
    i2 = w2*y1 + b2;
    y2 = sigmoid(i2);
  endfunction
  
  function [x] = sigmoid(z)
    x = 1 ./ (1 + exp(-z));
  endfunction

  function [x] = sigmoidPrime(z)
    x = z.*(1-z);
  endfunction
