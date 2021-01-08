function train(data, data_teste)
  for i = 1:3
    printf('T%d\n',i)
    for j = 1:3
      printf('TDNN%d\n',j)
      [w1,w2,eqm,epoch] = projeto_pratico_3(data,j);
      tmpdata = data(end-(j*5-1):end);
      out = projeto_pratico_3_teste(w1,w2,tmpdata',20);
      printf('%.4f\n',out')
      content_size = size(out,2);
      errm = sum(abs(data_teste-out')./data_teste)/content_size;
      printf('Erro relativo medio: %.4f\n', errm * 100)
      printf('Variancia: %.4f \n', var(out) * 100)
    endfor
  endfor
endfunction
