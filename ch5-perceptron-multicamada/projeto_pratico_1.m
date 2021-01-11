  function [w1, w2, eqm_curr,epoch, yy2] = projeto_pratico_1(X,d)
    epsilon = 10^-6;
    min = -2.4;
    max = 2.4;
    n = 0.1;
    epoch = 0;
    eqm_bef = 0;
    eqm_curr = 1;
    amount = size(X,1);
    w1 = (min + (max-min) * rand(10,3))/amount;
    w2 = (min + (max-min) * rand(1,10))/amount;
    b1 = rand();
    b2 = rand();
    
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy2 = [];
      for row = 1:amount
        xk = X(row,:);
        dk = d(row,:);
        [i1,i2,y1,y2] = forward(w1,w2,b1,b2,xk);
        yy2 = [yy2 y2];
        
        sig2 = -(dk'-y2).*sigmoidPrime(y2);
        w2 = w2 - n * sig2 * y1';
  
        sig1 = (w2'*sig2).*sigmoidPrime(y1);
        w1 = w1 - n * sig1 * xk;
      endfor
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
