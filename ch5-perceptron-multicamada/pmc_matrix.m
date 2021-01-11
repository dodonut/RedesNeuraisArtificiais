  function [yy3,epoch] = pmc_matrix(X,d)
    epsilon = 0.0000001;
    n = 0.05;
    epoch = 0;
    eqm_bef = 0;
    eqm_curr = 1;
    qtd_samples = size(X,1);
    dim_sample = size(X,2);
##    pesos definidos manualmente para testar as otimizacoes do backpropation
    w1 = rand(3,dim_sample);
    w2 = rand(3,dim_sample);
    w3 = rand(2,dim_sample);
    b1 = rand(3,qtd_samples);
    b2 = rand(3,qtd_samples);
    b3 = rand(2,qtd_samples); 
    
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy3 = [];
      
      [i1,i2,i3,y1,y2,y3] = forward(w1,w2,w3,b1,b2,b3,X);
      yy3 = [yy3 y3];
      
      sig3 = -(d'-y3).*tanhPrime(y3);
      w3 = w3 - n * sig3 * y2';
      b3 = b3 - n * b3 .* sig3;
      
      sig2 = (w3'*sig3).*tanhPrime(y2);
      w2 = w2 - n * sig2 * y1';
      b2 = b2 - n * b2 .* sig2;
      
      sig1 = (w2'*sig2).*tanhPrime(y1);
      w1 = w1 - n * sig1 * X;
      b1 = b1 - n * b1 .* sig1;
        
      epoch = epoch + 1;
      eqm_curr = eqm(yy3,d,4);
      if mod(epoch,500) == 0
        epoch
        eqm(yy3, d, amount)
      endif
    endwhile
  endfunction
  
  function [x] = eqm(Y,d,amount)
    yy = Y';
    x = 0;
    for i = 1:amount
      yk = yy(i,:);
      dk = d(i,:);
      p = power(dk-yk,2)/2;
      x = x + sum(p);
    endfor
    x = x/amount;
  
  endfunction
  
  function [i1,i2,i3,y1,y2,y3] = forward(w1,w2,w3,b1,b2,b3,X)
    i1 = w1 * X' + b1;
    y1 = tanh(i1);
    i2 = w2 * y1 + b2;
    y2 = tanh(i2);
    i3 = w3 * y2 + b3;
    y3 = tanh(i3);
  endfunction
  
  function [x] = tanhPrime(z)
    x = 1.0 - power(tanh(z),2);
  endfunction
