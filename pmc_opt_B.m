  function [yy3,epoch] = pmc_opt_B(X,d)
    epsilon = 0.0000001;
    epoch = 0;
    n = 0.05;
    eqm_bef = 0;
    eqm_curr = 1;
    amount = size(X,1);
##    pesos definidos manualmente para testar as otimizacoes do backpropation
    w1 = [0.464150   0.524372   0.607614
     0.484655   0.342216   0.014019
     0.976615   0.952809   0.751349];
    w2 = [0.196029   0.114800   0.129363
     0.780361   0.587616   0.091267
     0.475346   0.037567   0.412591];
    w3 = [0.41955   0.72192   0.78918
     0.80975   0.60164   0.70179];
     b1 =  0.13886;
     b2 =  0.13678;
     b3 =  0.72797;
     
     n1 = rand(3,3);
     n2 = rand(3,3);
     n3 = rand(2,3);
     
    err_ant1 = 0 * rand(3,3);
    err_ant2 = err_ant1;    
    err_ant3 = 0 * rand(2,3);
    err_curr1 = err_ant1;
    err_curr2 = err_ant1;
    err_curr3 = err_ant3;
    while abs(eqm_curr - eqm_bef) > epsilon
      eqm_bef = eqm_curr;
      yy3 = [];
      for row = 1:amount
        xk = X(row,:);
        dk = d(row,:);
        [i1,i2,i3,y1,y2,y3] = forward(w1,w2,w3,b1,b2,b3,xk);
        yy3 = [yy3 y3];
        to_execute = epoch > 0 || row > 1;
        
        sig3 = -(dk'-y3).*tanhPrime(y3);
        err_ant3 = err_curr3;
        err_curr3 = n * (sig3 * y2');
        dd = w3 - err_curr3;
        [dd, n3] = redefine_n(n3, dd, err_curr3, err_ant3, to_execute);
        old_w3 = w3;
        w3 = dd;
        
        
        sig2 = (w3'*sig3).*tanhPrime(y2);
        err_ant2 = err_curr2;
        err_curr2 = n * sig2 * y1';
        dd = w2 - err_curr2;
        [dd, n2] = redefine_n(n2,dd, err_curr2, err_ant2, to_execute);
        old_w2 = w2;
        w2 = dd;

  
        sig1 = (w2'*sig2).*tanhPrime(y1);
        err_ant1 = err_curr1;
        err_curr1 = n * sig1 * xk;
        dd = w1 - err_curr1;
        [dd,n1] = redefine_n(n1,dd, err_curr1, err_ant1,to_execute);
        old_w1 = w1;
        w1 = dd;

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
  
  function [new_dd, new_n] = redefine_n(n,w, err_curr, err_ant, to)
    new_n = n;
    new_dd = w;
    if to == false
      return
    endif
    new_n = n;
    d = sign(err_curr .* err_ant);
    c = -sign(err_curr);
    for i = 1:size(err_curr,1)
      for j = 1:size(err_curr,2)
        if d(i,j) > 0
          new_n(i,j) = 1.2 * n(i,j);
        elseif d(i,j) < 0
          new_n(i,j) = 0.5 * n(i,j);
        else
          new_n(i,j) = n(i,j);
        endif
      endfor
    endfor
    new_n = new_n .* c;
    new_dd = new_dd + new_n;  
  endfunction
  

  
  
  
  
  
  
  
  
  
  
  
  
