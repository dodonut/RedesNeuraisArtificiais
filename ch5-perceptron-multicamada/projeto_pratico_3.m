  function [w1, w2, eqm_curr,epoch] = projeto_pratico_3(data, tdnn)
    [X,d,w1,w2] = pre_process(data', tdnn);
    b1 = rand();
    b2 = rand();
    epsilon = 0.5*10^-6;
    n = 0.1;
    epoch = 0;
    eqm_bef = 0;
    eqm_curr = 1;
    amount = size(X,1);
    a = 0.9;
    old_w2 = w2;
    old_w1 = w1;
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy2 = [];
      for row = 1:amount
        xk = X(row,:);
        dk = d(row,:);
        [i1,i2,y1,y2] = forward(w1,w2,b1,b2,xk);
        yy2 = [yy2 y2];
        
        sig2 = -(dk'-y2).*sigmoidPrime(y2);
        tmpw2 = w2 + a * (w2 - old_w2) - n * sig2 * y1';
        old_w2 = w2;
        w2 = tmpw2;
  
        sig1 = (w2'*sig2).*sigmoidPrime(y1);
        tmpw1 = w1 + a * (w1 - old_w1) - n * sig1 * xk;
        old_w1 = w1;
        w1 = tmpw1;
        
      endfor
      epoch = epoch + 1;
      eqm_curr = eqm(yy2,d,amount);
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
  
  function [X,d,w1,w2] = pre_process(data, tdnn)
    ini = 0;
    p = 0;
    n = 0;
    X = [];
    d = [];
    if tdnn == 1
      p = 5;
      n = 10;
    elseif tdnn == 2
      p = 10;
      n = 15;
    else 
      p = 15;
      n = 25;
    endif
    t = size(data,2);
    s = t-p; 
    ini = p; 
    for i = 1:s
      tmpX = data(i:p);
      X = [X; tmpX];
      tmpd = data(p+1);
      d = [d; tmpd];
      p = p+1;
    endfor
    minimum = -2.4;
    maximum = 2.4;
    w1 = (minimum + (maximum-minimum) * rand(n,ini))/t;
    w2 = (minimum + (maximum-minimum) * rand(1,n))/t;
  endfunction
  
