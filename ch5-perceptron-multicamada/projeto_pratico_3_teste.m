function [yy2] = projeto_pratico_3_teste(w1,w2,X,size_predict)
  yy2 = [];
  for row=1:size_predict
    [i1,i2,y1,y2] = forward(w1,w2,X);
    yy2 = [yy2 y2];
    X = [X(2:end) y2];
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