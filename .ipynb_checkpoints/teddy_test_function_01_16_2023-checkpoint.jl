################################# Functions used for all parameters in the function called my_function() #################################

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)
#This function is used in the function called my_function(), in order to load and train the data of the neural network
#it returns the model and resDict 
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
#####################################################################################################################################################################

#################################################### Functions used when parameter = 1 ####################################################

#parameter = 1 means that: We drop the last n rows and columns of the adjacency matrix. Then the new S matrix is calculated. Same number of rows from x matrix are also dropped.


##This function reduces the size of a matrix by dropping the last n_col_row_to_reduce rows-columns. 
#It uses the function A2S in order to convert the reduced adjacency matrix to S matrix. 
#It returns the final matrix in onecold form
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
#####################################################################################################################################################################

#################################################### Functions used when parameter = 2 ####################################################
#parameter = 2 means that: adjacency matrix is splited in smaller matrices with size (size(adj_mtrx, 1)/ n_lines_to_split, size(adj_mtrx, 1)/ n_lines_to_split) and each smaller #matrix is saved in a dictionary. Then the x matrix is splited accordingly and each smaller matrix is saved in another dictionary.
#Then the dictionary with the splited adjacency matrix is converted to S matrix and then is multiplied with the dictionary containing the #splited x matrices. Then this dictionary is converted to matrix and multiplied with weight function



#This function gets the divisors of a number
#It is used when parameter = 2 in order to get the n_lines_to_split (mentioned on the above), in order to split the adjacency matrix. Code #uses get_divisors(size(adj_matrix, 2))[2] because 1 gives the original adjacency matrix, so 2 is the next one. We can definately use other numbers such as 3 or 4. Example, if adjacency matrix is 10 x 10 then get_divisors() will return 1 2 5 10. If [1] is used, then adj_matrix will break in  10/1 x 10/1 = 10 x10 matrix (same as previousely). If [2] is selected, then adj_matrix will break as  10/2 x 10/2 = 5 x 5 matrix. So the adj_matrix will be splited in four 5x5 matrices.

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


#This function splits the adjacency matrix  matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
#adj_mtrx = the adj matrix
#size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
#The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
#This function is used in the function my_function() in when the parameter is equal to 2. 
#It returns the adjacency matrix splited as a dictionary
#it is used in the function multiply_splited_2() in order to split the adjacency matrix which is given

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
 #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
 #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
 #This function is used in the function my_function() in when the parameter is equal to 2. 
 #It is the same with the function split_adj_matrix() but it splits the x matix
 #It returns the x matrix splited as a dictionary
 #it is used in the function multiply_splited_2() in order to split the x matrix which is given
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
#size(adj_mtrx, 1)/ n_lines_to_split =  is the number of rows, columns the small matrix will have.
#x_splited_length = the length of x_splited matrix, which is a dictionary
#It returns the s_matrix*x_matrix in a matrix form
#This matrix returns the 
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

#This function multiplies the splited matrices which are saved in dictionary form and converts it to matrix form.
#a_mtrx = adjacency matrix
#x_mtrx = x matrix
#th_mtrx = weight matrix 
#size(adj_mtrx, 1)/ n_lines_to_split = is the number of rows, columns the small matrix will have.
#First it calles the functions  split_adj_matrix() and split_x_matrix() in order to split the adjacency matrix and x matrix and get them in #dictionay form.
#Then it converts the adjacency matrix to S matrix in a dictionary form, using the splited adjacency matrix
#Then it multiplies the dictionaries with the splited S matrix and x matrix. This is saved in a dictionary
#Then it calls the function s_matrix_x_xmtrx_dictionary_to_matrix() in order to connvert it to matrix form

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
#####################################################################################################################################################################

#################################################### Functions used when parameter = 3 ####################################################
#parameter = 3 means that: 10% of the adjacency matrix is reduced. This 10% is controled by the factor 0.1 in the for loop when parameter=3
#Radomly the i row and the i column of the adjacency matrix are droped. The same i row is droped from the x matrix and the same i row is droped from the Ydata, which is 0 and 1.  

#There are no functions for parameter = 3, however there are some simple comands which do what is described on the above. 
#####################################################################################################################################################################



############################################################ After parameters ############################################################
#After we choose which method to use for spliting the matrices, we load the data in the function load_and_train_model() to get the weight matrix. We multiply the weight with the s_matrix * x_matrix. Then we get the onecold(s_matrix * x_matrix) to compare with our initial data.

#####################################################################################################################################################################





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
    final_matrix = multiply_splited_2(adj_matrix, x_matrix, weight, Int(n_lines_to_split))
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
