  function [yy3,epoch] = pmc(X,d)
    epsilon = 0.0000001;
    n = 0.05;
    epoch = 0;
    eqm_bef = 0;
    eqm_curr = 1;
    amount = size(X,1);
##    pesos definidos manualmente para testar as otimizacoes do backpropation
##    w1 = [0.464150   0.524372   0.607614
##     0.484655   0.342216   0.014019
##     0.976615   0.952809   0.751349];
##    w2 = [0.196029   0.114800   0.129363
##     0.780361   0.587616   0.091267
##     0.475346   0.037567   0.412591];
##    w3 = [0.41955   0.72192   0.78918
##     0.80975   0.60164   0.70179];
     w1 = rand(3,3);
     w2 = rand(3,3);
     w3 = rand(2,3);
     b1 =  rand();
     b2 =  rand();
     b3 =  rand();
    
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy3 = [];
      for row = 1:amount
        xk = X(row,:);
        dk = d(row,:);
        [i1,i2,i3,y1,y2,y3] = forward(w1,w2,w3,b1,b2,b3,xk);
        yy3 = [yy3 y3];
        
        sig3 = -(dk'-y3).*tanhPrime(y3);
        w3 = w3 - n * sig3 * y2';
        
        sig2 = (w3'*sig3).*tanhPrime(y2);
        w2 = w2 - n * sig2 * y1';
  
        sig1 = (w2'*sig2).*tanhPrime(y1);
        w1 = w1 - n * sig1 * xk;
      endfor
      epoch = epoch + 1;
      eqm_curr = eqm(yy3,d,4);
      if mod(epoch,500) == 0
        epoch
        eqm(yy3, d, amount)
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
  
  function [i1,i2,i3,y1,y2,y3] = forward(w1,w2,w3,b1,b2,b3,x)
    i1 = w1*x' + b1;
    y1 = tanh(i1);
    i2 = w2*y1 + b2;
    y2 = tanh(i2);
    i3 = w3*y2 + b3;
    y3 = tanh(i3);
  endfunction
  
  function [x] = tanhPrime(z)
    x = 1.0 - power(tanh(z),2);
  endfunction
