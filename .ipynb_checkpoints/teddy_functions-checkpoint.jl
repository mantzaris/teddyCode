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
