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
  import Pkg; Pkg.add("InvertedIndices") #install InvertedIndices in GoogleColab, you might not need that
  using InvertedIndices #use this in the begining of the code, importing it here causes a problem in GoogleColab, maybe it's a bug
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

function total_error(n_drop_row_col, a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim )
  mean_average_error = Vector{Float64}()
  a = initial_matrix(a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[1]
  aj = initial_matrix(a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[2]
  x = initial_matrix(a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[3]
  th = initial_matrix(a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[4]
  Final = initial_matrix(a_dims, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[5]
  for i in range(-1, n_drop_row_col)
   F1 = reduce_size(a, x, th, i)
   MAE = mae(Final, F1)
   append!(mean_average_error,MAE)
  end
  skip = collect(-1: n_drop_row_col)
  return mean_average_error, skip
  end


function split_matrix(a_mtrx, n_lines_to_split)
  x = size(a_mtrx, 1)
  y = size(a_mtrx, 2)
  n = div(x, n_lines_to_split)
  dictionary = Dict()
  count = 1
  k = 1
  for i in range(1, n_lines_to_split)
    z = 1
    for j in range(1, n_lines_to_split)
      dictionary["a_$count"] = a_mtrx[k: k + n - 1, z: j*n ]
      count = count + 1
      z = z + n
    end
    k = k + n
  end
  #show(stdout, "text/plain", dictionary["a_7"])
  return(dictionary)
end


function multiply_splited(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim, n_lines_to_split)
  a_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[1]
  aj_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[2]
  x_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[3]
  th_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[4]
  final_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[5]

  a_splited = split_alpha_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  aj_splited = Dict()
  print("\n")
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  print("\n")
  #show(stdout, "text/plain", aj_splited)
  final_mtrx_splited = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      final_mtrx_splited["Fin_$k"] = a_splited["a_$k"] * x_splited["x_$i"] * th
      k = k + 1
    end
  end
return(final_mtrx_splited)
end


function dictionary_to_matrix(dict, x_splited_length, x_y_dim, x_mtrx, n_lines_to_split)
  i = 1 
  mtrx_reduced_dictionary = Dict()
  count = 1
  num = div(size(x_mtrx, 1), n_lines_to_split)
  while (i <= length(dict)) 
    s = zeros(num, x_y_dim)
    k = 1 
    while k <= x_splited_length  
      s = s + dict["aj_x_$i"] 
      k = k + 1 
      i = i + 1
    end 
  mtrx_reduced_dictionary["A_$count"] = s
  count = count + 1
  end 
  s_ = vcat(mtrx_reduced_dictionary["A_1"])
  for i in range(2, length(mtrx_reduced_dictionary))
    aux = vcat(s_, mtrx_reduced_dictionary["A_$i"])
    s_ = aux
  end
  return s_ 
end 


function multiply_splited(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim, n_lines_to_split)
  a_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[1]
  aj_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[2]
  x_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[3]
  th_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[4]
  final_mtrx = initial_matrix(a_dim, a_down_lim, a_up_lim, x_y_dim, x_down_lim, x_up_lim, th_y_dim, th_down_lim, th_up_lim)[5]

  a_splited = split_alpha_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  aj_splited = Dict()
  print("\n")
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  #show(stdout, "text/plain", aj_splited)
  aj_x = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      aj_x["aj_x_$k"] = aj_splited["aj_$k"] * x_splited["x_$i"] 
      k = k + 1
    end
  end
  #for i in range(1, length(aj_x))
  #  show(stdout, "text/plain", aj_x["aj_x_$i"])
  #  print("\n")
  #end
  aj_x_mtrx = dictionary_to_matrix(aj_x, length(x_splited), x_y_dim, x_mtrx, n_lines_to_split)
  print("\n")
  show(stdout, "text/plain", aj_x_mtrx)
  return(aj_x_mtrx)
end
