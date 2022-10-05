#functions 

#get_row_columns_element
function get_each_element(mtrx, dim)
  for i in range(1, dim)
    for j in range(1, dim)
     print("mtrx[$i, $j]  = ", mtrx[i, j], "\n")
    end
  end
end


function make_mtrx(dimentions, list_1, list_2)
  m = zeros(dimentions, dimentions)
  for i in list_1
    for j in list_2
      m[i, j] = 1
    end
  end
  return m
end


function get_zeros_non_zeros(mtrx, x_dim, y_dim)
  s = 0
    for i in range(1, x_dim)
      for j in range(1, y_dim)
        if (a[i, j] == 1.0)
          s = s + 1
        end
      end
    end
    print("Non zeros: $s and zeros: ", x_dim * y_dim - s)
  return s
end

function random_matrix(a, b, x_dim, y_dim)
  Random.seed!(1234)
  mtrx = rand(a:b, x_dim, y_dim )
  print(mtrx)
  return mtrx
end


function initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)
  a = random_matrix(a_down_lim, a_up_lim, a_dim, a_dim)
  aj = A2S(a)
  x = random_matrix(x_down_lim, x_up_lim, a_dim, x_y_dim)
  th = random_matrix(th_down_lim, th_up_lim, x_y_dim, th_y_dim)
  Final = aj * x * th
  return a, aj, x, th, Final
end


function reduce_size(a, x, th, n_col_row_to_reduce)
  x_dim = size(a, 1)
  y_dim = size(a, 2)
  a_reduced = a[Not(x_dim - n_col_row_to_reduce: end), Not(y_dim - n_col_row_to_reduce : end)]
  aj_reduced = A2S(a_reduced)
  x_reduced = x[Not(x_dim - n_col_row_to_reduce: end), :]
  Final_reduced = aj_reduced * x_reduced * th
  return Final_reduced
  end


function mae(mtrx, mtrx_reduced)
  x = size(mtrx_reduced, 1)
  y = size(mtrx_reduced, 2)
  s = 0
  for i in range(1, x)
    for j in range(1, y)
      s = s + abs(mtrx[i, j] - mtrx_reduced[i, j])
    end
  end
  mae = sqrt(s/(x*y))
  println("MAE = ", mae)
  return mae 
  end
