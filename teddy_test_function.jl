#This function gets the divisors of a number
function get_divisors(number)
  divisors = Vector{Float64}()
  for i in range(1, div(number, 2))
    if number % i == 0
      append!(divisors, Int(i))
    end
  end
  append!(divisors, Int(number))
  return divisors
end

##This function reduces the size of a matrix by throughing the last n_col_row_to_reduce rows-columns
#adj = aj matrix
#x = xmatrix 
#th = th_matrix, wheight matrix 
#n_col_row_to_reduce = how many rows-columns we want to drop

function reduce_size(adj, x, th, n_col_row_to_reduce)
  x_dim = size(adj, 1)
  y_dim = size(adj, 2)
  k = 3
  adj_reduced = adj[Not(x_dim - n_col_row_to_reduce + 1: end), Not(y_dim - n_col_row_to_reduce + 1: end)]
  s_reduced = A2S(adj_reduced)
  x_reduced = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
  Final_reduced = s_reduced^k * x_reduced * th'
  return onecold(Final_reduced')
  end

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)

function load_and_train_model(SX_, yhot_)
  resDict_ = Dict()
  model_ = Chain( Dense( size(SX_, 1) => size(yhot_, 1)) , softmax)
  loss(x, y) = Flux.crossentropy(model_(x), y)
  opt = Adam(0.01)
  pars = Flux.params(model_)
  data = Flux.DataLoader((SX_, yhot_) , batchsize = 10 , shuffle = true)
  epochs_ = Int64[]
  for epoch in 1:500
    Flux.train!(loss, pars, data ,opt)
    push!(epochs_, epoch)
  end 
  resDict_["params"] = pars
  resDict_["model"] = model_
  return(model_, resDict_)
end 



#This function splits the adjacency matrix  matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
 #adj_mtrx = the adj matrix
 #n_lines_to_split is the number of rows, columns the small matrix will have.
 #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.

 function split_adj_matrix(adj_mtrx, n_lines_to_split)
  x = size(adj_mtrx, 1)
  y = size(adj_mtrx, 2)
  n = div(x, n_lines_to_split)
  dictionary = Dict()
  count = 1
  k = 1
  for i in range(1, n_lines_to_split)
    z = 1
    for j in range(1, n_lines_to_split)
      dictionary["a_$count"] = adj_mtrx[k: k + n - 1, z: j*n ]
      count = count + 1
      z = z + n
    end
    k = k + n
  end
  return(dictionary)
end


#This function splits the x matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
 #a_mtrx = the a_matrix
 #n_lines_to_split is the number of rows, columns the small matrix will have.
 #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
  
  function split_x_matrix(x_mtrx, n_lines_to_split)
  x = size(x_mtrx, 1)
  n = div(x, n_lines_to_split)
  #print(n)
  dictionary = Dict()
  z = 1
  for i in range(1, n_lines_to_split)
    dictionary["x_$i"] = x_mtrx[z : z + n - 1 , 1:end]
    z = z + n
  end
  return(dictionary)
  end


#This matrix turns s_matrix*x_matrix (in dictionary form), to matrix
#dict = the dictionary we want to convert to matrix
#x_mtrx = x_matrix
#n_lines_to_split =  is the number of rows, columns the small matrix will have.
#It returns the s_matrix*x_matrix in a matrix form

function s_matrix_x_xmtrx_dictionary_to_matrix(dict, x_splited_length, x_mtrx, n_lines_to_split)
  i = 1 
  mtrx_reduced_dictionary = Dict()
  count = 1
  num = div(size(x_mtrx, 1), n_lines_to_split)
  while (i <= length(dict)) 
    x_y_dim = size(x_mtrx, 2)
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

function multiply_splited_2(a_mtrx, x_mtrx, th_mtrx, n_lines_to_split)
  k = 3
  a_splited = split_adj_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  #x_y_dim = size(x_mtrx, 2)
  aj_splited = Dict()
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  aj_x = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      aj_x["aj_x_$k"] = aj_splited["aj_$k"]^3 * x_splited["x_$i"] 
      k = k + 1
    end
  end
  aj_x_mtrx = s_matrix_x_xmtrx_dictionary_to_matrix(aj_x, length(x_splited), x_mtrx, n_lines_to_split)'
  final = (th_mtrx * aj_x_mtrx )
  return(onecold(final))
end


#adj_matrix = adjacency matrix
#x_matrix = x matrix
#y_matrix = y matrix (actual data)
#paremeter = says which method will be used to split the adj_matrix 

function my_function(adj_matrix, x_matrix, y_matrix, parameter)
  #get Y' data by training the network, before seting the parameter
  k = 3
  yhot = onehotbatch(vec(y_matrix), [1, 2])
  S = A2S(adj_matrix)
  SX = S^k * x_matrix
  SX = SX'   
  train_x = SX
  train_y = yhot
  #show(stdout, "text/plain", SX_TH ) 
  if parameter == 1
    yhot = onehotbatch(vec(y_matrix), [1, 2])
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    SX = SX'   
    train_x = SX
    train_y = yhot
    model, resDict = load_and_train_model(SX, yhot)
    weight = resDict["model"].layers[1].weight
    accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
    println("Accuracy by training all data: ", accuracy, "%", ", k = $k")  
    test = weight * SX
    SX_TH = onecold(test, [1, 2])
    skiped = round(size(adj_matrix, 1) * 0.25, digits = 0)
    final_matrix = reduce_size(adj_matrix ,x_matrix, weight, skiped)
  elseif parameter == 2
    yhot = onehotbatch(vec(y_matrix), [1, 2])
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    SX = SX'   
    train_x = SX
    train_y = yhot
    model, resDict = load_and_train_model(SX, yhot)
    weight = resDict["model"].layers[1].weight
    accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
    println("Accuracy by training all data: ", accuracy, "%", ", k = $k")  
    test = weight * SX
    SX_TH = onecold(test, [1, 2])
    n_lines_to_split = get_divisors(size(adj_matrix, 2))[2]
    final_matrix = multiply_splited_2(adj_matrix, x_mtrx, weight_mtrx, Int(n_lines_to_split))
  elseif parameter == 3 
    for i in range(1, Int(size(adj_matrix, 1) * 0.1))
      random_number = rand(1: size(adj_matrix, 1))
      adj_matrix = adj_matrix[1:end .!= random_number, 1:end .!= random_number]
      x_matrix = x_matrix[1:end .!= random_number, 1:end]
      y_matrix = y_matrix[1:end .!= random_number, 1:end]
  end
    yhot = onehotbatch(vec(y_matrix), [1, 2])
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    SX = SX'   
    train_x = SX
    train_y = yhot
    model, resDict = load_and_train_model(SX, yhot)
    weight = resDict["model"].layers[1].weight
    accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
    println("Accuracy by training all data: ", accuracy, "%", ", k = $k")
    println("SX: ", size(SX))
    println("weight: ", size(weight))
    final_matrix_aux = weight * SX 
    final_matrix = onecold(final_matrix_aux)
  end 
  return(final_matrix)
end 
