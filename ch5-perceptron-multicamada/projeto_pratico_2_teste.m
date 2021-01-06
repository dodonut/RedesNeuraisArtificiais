function [yy2] = projeto_pratico_2_teste(w1,w2,X,d)
  amount = size(X,1);
  yy2 = [];
  for row=1:amount
    xk = X(row,:);
    dk = d(row,:);
    [i1,i2,y1,y2] = forward(w1,w2,xk);
    yy2 = [yy2 y2];
  endfor
endfunction

function [i1,i2,y1,y2] = forward(w1,w2,x)
  i1 = w1*x';
  y1 = sigmoid(i1);
  i2 = w2*y1;
  y2 = sigmoid(i2);
endfunction

function [x] = sigmoid(z)
  x = 1 ./ (1 + exp(-z));
endfunction