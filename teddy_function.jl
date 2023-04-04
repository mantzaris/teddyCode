using Plots
using Distributions
using StatsBase
using InvertedIndices
using Flux 
using OneHotArrays
using Graphs
using GraphPlot
using Distributions
using LinearAlgebra
using Random 
using InvertedIndices
<<<<<<< HEAD


################################# Functions used for all parameters in the function called my_function() #################################



function split_train_test(Xdata, Ydata, percentage_to_train)
    #sample function gets numbers without repeating them
    number_of_rows = Int(round(percentage_to_train * size(Xdata, 1), digits = 0))
    train_id = sort(sample(1: size(Xdata, 1), number_of_rows, replace = false))
    test_id = (1:size(Xdata, 1))[Not(train_id)]
    trainX = Xdata[train_id, :]
    testX = Xdata[test_id, :]
    trainY = Ydata[train_id, :]
    testY = Ydata[test_id, :]
    return trainX, testX, trainY, testY
  end
  
  
  
  #This function loads the data and trains the neural network. 
=======
using MLDatasets: Cora


 ##################################################### generate data to test the my_function #####################################################


#This function loads the data and trains the neural network. 
>>>>>>> teddy
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
    loss_ = Float64[]
    for epoch in 1:500
      Flux.train!(loss, pars, data ,opt)
      push!(epochs_, epoch)
      push!(loss_, loss(SX_, yhot_))
    end 
    resDict_["params"] = pars
    resDict_["model"] = model_
    return(model_, resDict_, epochs_, loss_)
  end 

<<<<<<< HEAD
=======
  #convert adj to S marix
  function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end

#This function makes data using Normal or Uniform distribution
function create_adjacency_matrix_normal_uniform(num, distro1)
  Random.seed!(10)
  ER_tmp = erdos_renyi( num , 10*(num) )
  BA_tmp = barabasi_albert( num , 8 )
  SF_tmp = static_scale_free( num , 8*(num) , 4 )
  WS_tmp = erdos_renyi( num , 10*(num) ) #barabasi_albert( NN_tmp , 5 )
  blocks_tmp = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
    
  #now add some edges between the blocks that are the communities
  for bb in 1:Int(round(num/10))
    for b1 in 0:3
      for b2 in 0:3
        if(b1 < b2)
          range1 = randperm(num)[1] + b1*num
          range2 = randperm(num)[1] + b2*num
          add_edge!( blocks_tmp , range1 , range2 )
        end
      end
    end
  end
  density_nn = Graphs.density(blocks_tmp)
  adj = Matrix(adjacency_matrix(blocks_tmp))

 
  d1 = rand(distro1(0, 1), 3 * num)
  d2 = rand(distro1(0, 1), 3 * num)
  d3 = rand(distro1(0, 1), 3 * num)
  c1 = Categorical( [0.5,0.25,0.25] )
  c2 = Categorical( [0.15,0.15,0.7] )
  c3 = Categorical( [0.5,0.5,0] )

 

   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1a = vcat( xd1 , xc1 )'
   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1b = vcat( xd1 , xc1 )'
   xd2 = rand( d2 ,  3 * num )
   xd2 = reshape(xd2, 3 ,num)
   xc2 = onehotbatch( rand( c2 , num ) , 1:3 )
   x2 = vcat( xd2 , xc2 )'
   xd3 = rand( d3 ,  3 * num )
   xd3 = reshape(xd3, 3 ,num)
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'

  X = vcat( x1a , x1b , x2 , x3 )
  y1a = onehotbatch( 1*ones(num) , 1:2 )'
  y1b = onehotbatch( 1*ones(num) , 1:2 )'
  y2 = onehotbatch( 2*ones(num) , 1:2 )'
  y3 = onehotbatch( 2*ones(num) , 1:2 )'
  Y = vcat(y1a, y1b, y2, y3)
  Y_to_use = vcat(1*ones(num), 1*ones(num), 2*ones(num), 2*ones(num))
  #println("size(X): ", size(X))
  #println("size(Y):", size(Y_to_use))
  return adj, X, Y_to_use
end 

#this function checks if I the y_hat data predicted by the network are the same with the y_hat_manually data, which is the data I predicted with the classical way: F(WX + bias)
#vec1 = y_hat
#vec2 = y_hat_manually

function check_validity(vec1, vec2)
  s = 0 
  for i in 1:length(vec1)
    if vec1[i] == vec2[i]
      s += 1
    end
  end
  if s == length(vec1)
    println("The two methods match")
  else
    println("Methods do not match")
  end 
end 

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)


function get_adj_x_weight(num, k)
  #num = 1000
  distro1 = Normal 
  #k = 3
  ad, x, y = create_adjacency_matrix_normal_uniform(num, distro1)
  
  yhot = onehotbatch(y, [1, 2])
  S = A2S(ad)
  SX = S^k * x
  SX
  SX = SX'
  # println("SX = ", SX)
  train_x = SX
  # println("train_x = ", train_x)
  train_y = yhot
  #println("Training with raw data")
  #epochs, loss_on_train, loss_on_test,  model, resDict = load_and_train_3(SX, yhot, train_x, train_y)
  model, resDict, epochs, loss = load_and_train_model(SX, yhot)
  weight = resDict["model"].layers[1].weight
  # println("weight = ", weight)
  accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
  println("Accuracy: ", accuracy, "%", ", k = $k, distribution = $distro1")      
  y_hat = onecold( model(train_x), [1, 2])
  y_actual = onecold(train_y, [1, 2])
  test = weight * SX
  #show(stdout, "text/plain", test)
  println("weight * SX = ", size(test))
  bias_ = resDict["model"].layers[1].bias
  bias_1 = fill(bias_[1], (1, size(test, 2)))
  bias_2 = fill(bias_[2], (1, size(test, 2)))
  bias = vcat(bias_1, bias_2)
  z = test + bias
  z_pred = softmax(z)
  y_hat_manually = onecold(softmax(z), [1, 2])
  #println("y_hat_manually = ", y_hat_manually)
  # check_validity(y_hat, y_hat_manually)
  SX_TH = onecold(test, [1, 2])
  #println("SX * Θ = ", SX_TH)
  #println("SX * Θ = ", y)
  #println(typeof(y))
  #show(stdout, "text/plain", test)
  y_ = Vector{Int64}()
  for i in 1:length(y)
    append!(y_, convert(Int64, y[i]))
  end  
  # println("SX * Θ_(no_onecold) = ", test)
  # println("SX * Θ  = ", SX_TH)
  # println("y_train = ", y_) #actual values
  s = 0
  for i in 1:length(y_)
    if y_[i] == SX_TH[i]
      s += 1
    end 
  end 
  ac = (s/length(y_))*100
  println("acccuracy SX * Θ = ", ac , "%")
  #show(stdout, "text/plain", S)
  #show(stdout, "text/plain", ad)
  #println(typeof(y))
  #println("weight * SX = ", test)
  return ad, x, weight, SX_TH
  end


  distro1 = Normal
  num = 1000
  d1 = rand(distro1(0, 1), 3 * num)
  d2 = rand(distro1(0, 1), 3 * num)
  d3 = rand(distro1(0, 1), 3 * num)
  c1 = Categorical( [0.5,0.25,0.25] )
  c2 = Categorical( [0.15,0.15,0.7] )
  c3 = Categorical( [0.5,0.5,0] )

 

   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1a = vcat( xd1 , xc1 )'
   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1b = vcat( xd1 , xc1 )'
   xd2 = rand( d2 ,  3 * num )
   xd2 = reshape(xd2, 3 ,num)
   xc2 = onehotbatch( rand( c2 , num ) , 1:3 )
   x2 = vcat( xd2 , xc2 )'
   xd3 = rand( d3 ,  3 * num )
   xd3 = reshape(xd3, 3 ,num)
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'

  X = vcat( x1a , x1b , x2 , x3 )
  y1a = onehotbatch( 1*ones(num) , 1:2 )'
  y1b = onehotbatch( 1*ones(num) , 1:2 )'
  y2 = onehotbatch( 2*ones(num) , 1:2 )'
  y3 = onehotbatch( 2*ones(num) , 1:2 )'
  Y = vcat(y1a, y1b, y2, y3)


  Y_to_use = vcat(1*ones(num), 1*ones(num), 2*ones(num), 2*ones(num))


##################################################################################################################################################
  #Data Generation
  adj_mtrx, x_mtrx, weight_mtrx, sx_theta = get_adj_x_weight(1000, 3)



################################# Functions used for all parameters in the function called my_function() #################################



function split_train_test(Xdata, Ydata, percentage_to_train)
    #sample function gets numbers without repeating them
    number_of_rows = Int(round(percentage_to_train * size(Xdata, 1), digits = 0))
    train_id = sort(sample(1: size(Xdata, 1), number_of_rows, replace = false))
    test_id = (1:size(Xdata, 1))[Not(train_id)]
    trainX = Xdata[train_id, :]
    testX = Xdata[test_id, :]
    trainY = Ydata[train_id, :]
    testY = Ydata[test_id, :]
    return trainX, testX, trainY, testY
  end
  
  
  
>>>>>>> teddy



  #################################################### Functions used when parameter = 1 ####################################################
  
  #parameter = 1 means that: We drop the last n rows and columns of the adjacency matrix. Then the new S matrix is calculated. Same number of rows from x matrix are also dropped.
  
  
  ##This function reduces the size of a matrix by dropping the last n_col_row_to_reduce rows-columns. 
  #It uses the function A2S in order to convert the reduced adjacency matrix to S matrix. 
  #It returns the final matrix in onecold form
  #adj = aj matrix
  #x = xmatrix 
  #th = th_matrix, wheight matrix 
  #n_col_row_to_reduce = how many rows-columns we want to drop
  
  function reduce_size(adj, x, y,  n_col_row_to_reduce)
    x_dim = size(adj, 1)
    y_dim = size(adj, 2)
    k = 3
    adj_reduced = adj[Not(x_dim - n_col_row_to_reduce + 1: end), Not(y_dim - n_col_row_to_reduce + 1: end)]
    s_reduced = A2S(adj_reduced)
<<<<<<< HEAD
    x_reduced = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    y_reduced_ = y[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    SX_reduced_ = s_reduced^k * x_reduced
    return SX_reduced_, y_reduced_
=======
    x_reduced_ = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    y_reduced_ = y[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    SX_reduced_ = s_reduced^k * x_reduced_
    return SX_reduced_, y_reduced_, x_reduced_
>>>>>>> teddy
    end

#This matrix calculates the accuracy between two matrices, the actual Ydata and the predicted_Ydata
function check_accuracy(Ydata, Ydata_predicted)
  s = 0
  for i in range(1, length(Ydata_predicted))
    if Ydata[i] == Ydata_predicted[i]
      s += 1
    end 
  end 
  accuracy = round(s/length(Ydata_predicted), digits = 2)
  println("Accyracy to get Ydata, using weight average matrix and SX i.e. (SX * weight_average_matrix): ", accuracy * 100, "%")
  end  


  
  #This matrix calculates the accuracy between two matrices, the actual Ydata and the predicted_Ydata
function check_accuracy2(Ydata, Ydata_predicted)
  s = 0
  for i in range(1, length(Ydata_predicted))
    if Ydata[i] == Ydata_predicted[i]
      s += 1
    end 
  end 
  accuracy = round(s/length(Ydata_predicted), digits = 2)
  println("Accyracy to get Ydata, using weight calculated on the above and SX i.e. (SX * weight_average_matrix): ", accuracy * 100, "%")
  end  



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
  

  #This function splits the y matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
   #y_mtrx = the Ydata
   #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.
   #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
   #This function is used in the function my_function() in when the parameter is equal to 2. 
   #It is the same with the function split_adj_matrix() but it splits the y matix
   #It returns the y matrix splited as a dictionary
   #it is used in the function multiply_splited_2() in order to split the Ydata matrix which is given
    function split_y_matrix(y_mtrx, n_lines_to_split)
    x = size(y_mtrx, 1)
    n = div(x, n_lines_to_split)
    dictionary = Dict()
    z = 1
    for i in range(1, n_lines_to_split)
      dictionary["y_$i"] = y_mtrx[z : z + n - 1 , 1:end]
      z = z + n
    end
    return(dictionary)
    end

  
  #This function creates the adjacency matrix times the x matrix in dictionary form
  #a_mtrx = adjacency matrix
  #x_mtrx = X matrix 
  #size(adj_mtrx, 1)/ n_lines_to_split is the number of rows, columns the small matrix will have.

  function adj_matrxix_x_X_matrix_dictionary(a_mtrx, x_mtrx, n_lines_to_split)
  k = 3
  a_splited = split_adj_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  #x_y_dim = size(x_mtrx, 2)
  aj_splited = Dict()
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  sx = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      sx["sx_$k"] = aj_splited["aj_$k"]^3 * x_splited["x_$i"] 
      k = k + 1
    end
  end
  return sx 
  end 
  
  

#This function combines the data we have. For example if SX_segmented = {"SX_1", "SX_2", "SX_3", "SX_4", "SX_5", "SX_6", "SX_7", "SX_8", "SX_9", "SX_10"} 
#and Ydata_segmented = {"Y1", "Y2"}, then this function returns this:
#[(SX_1, Y1), (SX_2, Y1), (SX_3, Y1), (SX_4, Y1), (SX_5, Y1), (SX_6, Y2), (SX_7, Y2), (SX_8, Y2), (SX_9, Y2), (SX_10, Y2) ], this if done for training purposes
#SX_segmented is a dictionary. SX_1, SX_2....are smaller matrices, so SX_segmented can be converted to a matrix [SX_1, SX_2, SX_3, SX_4, SX_5; SX_6, SX_7, SX_8, SX_9, SX_10] which is the original SX matrix

#SX_segmented_ = the segmented SX matrix in dictionary form
#Ydata_segmented_ = the segmented Ydata matrix in dictionary form
function combine_data(SX_segmented_, Ydata_segmented_)
  combined_data_ = []
  count = 1
  for i in range(1, length(Ydata_segmented_))
    for j in range(1, length(SX_segmented_) / length(Ydata_segmented_))
      push!(combined_data_, (SX_segmented_["sx_$count"], Ydata_segmented_["y_$i"]))
      count += 1
    end 
  end 
  return combined_data_ 
end 


#This function radomly reduces the ajacency matrix by p% percent
#adj_mtrx = adjacency matrix
#x_mtrx = the x matrix
#p is the percentage by which the adj_mtrx is being reduced, takes values from 0-1

  function radomly_reduced_matrix(adj_mtrx, x_mtrx, y_mtrx, p )
    k = 3
    for i in range(1, Int(size(adj_mtrx, 1) * p))
      random_number = rand(1: size(adj_mtrx, 1))
      adj_mtrx = adj_mtrx[1:end .!= random_number, 1:end .!= random_number]
      x_mtrx = x_mtrx[1:end .!= random_number, 1:end]
      y_mtrx = y_mtrx[1:end .!= random_number, 1:end]
    end
    S_reduced = A2S(adj_mtrx)
    SX_reduced_ = S_reduced^k * x_mtrx 
<<<<<<< HEAD
    y_reduced_ = y_mtrx
    return SX_reduced_, y_reduced_
=======
    x_reduced_ = x_mtrx
    y_reduced_ = y_mtrx
    return SX_reduced_, y_reduced_, x_reduced_
>>>>>>> teddy
  end 



#This function gets the SX times the weight matrix
#SX_ = is the S*X matrix
#weight_ = is the weight matrix
#y_matrix = the actual y data
function SX_x_Weight_matrix(SX_, weight_, y_matrix_)
  SX_x_weight_ = SX_ * weight_'
  predicted_mtrx_ = onecold( SX_x_weight_' )
  check_accuracy(y_matrix_, predicted_mtrx_)
  return predicted_mtrx_
end 


#This function gets the SX times the weight matrix
#SX_ = is the S*X matrix
#weight_ = is the weight matrix
#y_matrix = the actual y data
function SX_x_Weight_matrix2(SX_, weight_, y_matrix_)
  SX_x_weight_ = SX_ * weight_'
  predicted_mtrx_ = onecold( SX_x_weight_' )
  check_accuracy2(y_matrix_, predicted_mtrx_)
  return predicted_mtrx_
end 


#this functions creates a list of concecutive numbers that they will be dropped as rows and/or as columns
#mtrx = the matrix we want to reduce
#to_drop the number of rows or columns we want to drop
function to_drop(adj_, X_, Ydata_, to_skip)
  k = 3 # will be used for S matrix calculation
  Random.seed!(10)
  rows_drop = rand(1: size(adj_, 1) - to_skip + 1)
  for i in range(1, to_skip)
    adj_ = adj_[1:end .!= rows_drop, 1:end ]
    adj_ = adj_[1:end, 1:end .!= rows_drop]
    X_ = X_[1:end .!= rows_drop, 1:end ]
    Ydata_ = Ydata_[1:end .!= rows_drop, 1:end]
  end
  S = A2S(adj_)
  SX_reduced = S^k * X_
<<<<<<< HEAD
  return SX_reduced, Ydata_
=======
  return SX_reduced, Ydata_, X_
>>>>>>> teddy
end 



<<<<<<< HEAD
=======

function load_and_train_model_one_shot_learining(x_matrix, y_matrix, SX)
  testing_perc = 0.3
  targets = collect(1:2) #length(unique(y_matrix)) #collect(1:2) #2 = length(unique(y_matrix))
  error_dict = Dict()
  res_dict = Dict()
  stopInd = size(x_matrix)[1]
  testInds = Int.(round.( range(1, stop=stopInd, length=Int(round(testing_perc*stopInd))) ))
  trainInds = (1:stopInd)[Not(testInds)]
  X_dim =size(x_matrix')[1]
  Y_dim = length(unique(y_matrix))
  yhot = onehotbatch( y_matrix , collect(1:Y_dim) )
  modelSGC = Chain( Dense( X_dim => Y_dim ) , softmax )
  optim = Flux.Adam( 0.01 );
  pars = Flux.params( modelSGC )
  data = Flux.DataLoader( (SX', yhot) ,  batchsize=size(yhot)[2]  ,  shuffle=true )
  epoch_num = 1000
  errors = []
  for epoch in 1:epoch_num 
    for (x, y) in data
      val, grad = Flux.withgradient(pars) do
        Flux.crossentropy( modelSGC(x)[:,trainInds] , y[:,trainInds] )
      end 
    push!(errors, val)
    Flux.update!(optim, pars, grad)
    end
  end
  resDict = Dict()
  resDict["errors"] = errors
  resDict["training"] = mean( onecold( modelSGC(SX')[:,trainInds], targets ) .== y_matrix[trainInds] )
  resDict["testing"]  = mean( onecold( modelSGC(SX')[:,testInds], targets) .== y_matrix[testInds] )
  resDict["model"] = modelSGC
  weight = resDict["model"].layers[1].weight
  SX_weigth = SX * weight'
  predicted_mtrx = onecold( SX_weigth' )
  println("Acuuracy from training data: ", round(resDict["training"] * 100, digits = 4), "%")
  println("Acuuracy from testing data: ", round(resDict["testing"] * 100, digits = 4), "%")
  check_accuracy(y_matrix, predicted_mtrx)
end 


>>>>>>> teddy
  ########################### my_function modified ###########################

  #adj_matrix = adjacency matrix
  #x_matrix = x matrix
  #y_matrix = y matrix (actual data)
  #paremeter = says which method will be used to split the adj_matrix 
  
  function my_function(adj_matrix, x_matrix, y_matrix, parameter)
<<<<<<< HEAD
    #get Y' data by training the network, before seting the parameter
=======
>>>>>>> teddy
    k = 3
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    if parameter == 1
<<<<<<< HEAD
      a = 0.25 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced = reduce_size(adj_matrix ,x_matrix, y_matrix, skiped)
      #println("SX_reduced: ", size(SX_reduced))
      #println("y_reduced: ", size(y_reduced'))
      train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7)
      train_x = train_x'
      test_x = test_x'  
      yhot = onehotbatch(vec(train_y), [1, 2]) 
      train_y = yhot
      test_y = onehotbatch(vec(test_y), [1, 2])
      #println("train_x: ", size(train_x))
      #println("train_y: ", size(train_y))
      #println("test_x: ", size(test_x))
      #println("test_y: ", size(test_y))
      model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
      weight = resDict["model"].layers[1].weight
      accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
      #Get the predicted values by using SX matrix and our model
      #pred_by_model = onecold(model(SX'), [1, 2])
      #Get the predicted values by using the weight matrix

      #one function
      predicted_mtrx = SX_x_Weight_matrix(SX, weight, y_matrix)

      ############################## new part of the code ##############################
      weight_dict = Dict()
      m = 1
      #reducing the adjacency matrix by 1% 2% 3% 4% 5% 11% 12% 13% 14% 15% 20% and 25% and taking the average of weight matrices
      for i in [0.01, 0.02, 0.03, 0.04, 0.05, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.25]
        skiped = round(size(adj_matrix, 1) * i, digits = 0)
        println("skiped = ", skiped)
        SX_reduced, y_reduced = reduce_size(adj_matrix ,x_matrix, y_matrix, skiped)
        train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7)
        train_x = train_x'
        test_x = test_x'  
        yhot = onehotbatch(vec(train_y), [1, 2]) 
        train_y = yhot
        test_y = onehotbatch(vec(test_y), [1, 2])
        model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
        weight = resDict["model"].layers[1].weight
        weight_dict["w_$m"] = weight
        m += 1 
        accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
        println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
        SX_x_Weight_matrix2(SX, weight, y_matrix)
      end 
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum / length(weight_dict)
      predicted_mtrx = SX_x_Weight_matrix(SX, weight_average_matrix, y_matrix)
      SX_reduced_x_weight = SX_reduced * weight_average_matrix'
      predicted_mtrx_from_reduced = onecold( SX_reduced_x_weight' )
      s = 0
      for i in range(1, size(predicted_mtrx_from_reduced, 1))
        if predicted_mtrx_from_reduced[i] == y_reduced[i]
          s += 1
        end 
      end
      println("Accyracy to get reduced Ydata, using the weight average matrix and SX_reduced i.e. (SX_reduced * weight_average_matrix) is: ", round((s/length(predicted_mtrx)) * 100, digits = 2), "%")   
    elseif parameter == 2
      n_lines_to_split = get_divisors(size(adj_matrix, 2))[2]
      SX_segmented = adj_matrxix_x_X_matrix_dictionary(adj_matrix, x_matrix, Int(n_lines_to_split))
      Ydata_segmented = split_y_matrix(y_matrix, Int(n_lines_to_split))
      combined_data =  combine_data(SX_segmented, Ydata_segmented)
      weight_dict = Dict()
      for i in range(1, length(combined_data))
        train_x, test_x, train_y, test_y = split_train_test(combined_data[i][1] , combined_data[i][2], 0.7)
        train_x = train_x'
        test_x = test_x'  
        yhot = onehotbatch(vec(train_y), [1, 2]) 
        train_y = yhot
        test_y = onehotbatch(vec(test_y), [1, 2])
        #println("train_x: ", size(train_x))
        #println("train_y: ", size(train_y))
        #println("test_x: ", size(test_x))
        #println("test_y: ", size(test_y)) 
        model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
        weight = resDict["model"].layers[1].weight
        accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
        println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
        weight_dict["w_$i"] = weight
      end
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum/length(weight_dict)
      SX_x_weight = SX * weight_average_matrix'
      predicted_mtrx = onecold( SX_x_weight' )
      check_accuracy(y_matrix, predicted_mtrx)
    elseif parameter == 3 
      SX_reduced, y_reduced = radomly_reduced_matrix(adj_matrix ,x_matrix, y_matrix, 0.7 ); 
      train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7);
      train_x = train_x'
      test_x = test_x'  
      yhot = onehotbatch(vec(train_y), [1, 2]) 
      train_y = yhot
      test_y = onehotbatch(vec(test_y), [1, 2])
      model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
      weight = resDict["model"].layers[1].weight
      accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
      predicted_mtrx = SX_x_Weight_matrix2(SX, weight, y_matrix)
    elseif parameter == 4
      #p = 0.3
      #dropping = Int(size(adj_matrix, 1) * p)
      a = 0.25 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced = to_drop(adj_matrix, x_matrix, y_matrix, skiped)
      train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7)
      train_x = train_x'
      test_x = test_x'  
      yhot = onehotbatch(vec(train_y), [1, 2]) 
      train_y = yhot
      test_y = onehotbatch(vec(test_y), [1, 2])
      model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
      weight = resDict["model"].layers[1].weight
      accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
      println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
      predicted_mtrx = SX_x_Weight_matrix2(SX, weight, y_matrix)

      weight_dict = Dict()
      m = 1
      #reducing the adjacency matrix by 1% 2% 3% 4% 5% 11% 12% 13% 14% 15% 20% and 25% and taking the average of weight matrices
      for i in [0.01, 0.02, 0.03, 0.04, 0.05, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.25]
        skiped = round(size(adj_matrix, 1) * i, digits = 0)
        println("skiped = ", skiped)
        SX_reduced, y_reduced = to_drop(adj_matrix, x_matrix, y_matrix, skiped)
        train_x, test_x, train_y, test_y = split_train_test(SX_reduced , y_reduced, 0.7)
        train_x = train_x'
        test_x = test_x'  
        yhot = onehotbatch(vec(train_y), [1, 2]) 
        train_y = yhot
        test_y = onehotbatch(vec(test_y), [1, 2])
        model, resDict, epochs, loss = load_and_train_model(train_x, train_y)
        weight = resDict["model"].layers[1].weight
        weight_dict["w_$m"] = weight
        m += 1 
        accuracy = round(mean( onecold( model(test_x), [1, 2] ) .== onecold(test_y, [1, 2]) ) * 100, digits = 2)
        println("Accuracy by training tested data: ", accuracy, "%", ", k = $k")
        SX_x_Weight_matrix2(SX, weight, y_matrix)
      end 
      weight_sum = zeros(size(weight_dict["w_1"], 1), size(weight_dict["w_1"], 2))
      for i in range(1, length(weight_dict))
        weight_sum +=  weight_dict["w_$i"]
      end 
      weight_average_matrix = weight_sum / length(weight_dict)
      predicted_mtrx = SX_x_Weight_matrix(SX, weight_average_matrix, y_matrix)
      SX_reduced_x_weight = SX_reduced * weight_average_matrix'
      predicted_mtrx_from_reduced = onecold( SX_reduced_x_weight' )
      s = 0
      for i in range(1, size(predicted_mtrx_from_reduced, 1))
        if predicted_mtrx_from_reduced[i] == y_reduced[i]
          s += 1
        end 
      end
      println("Accyracy to get reduced Ydata, using the weight average matrix and SX_reduced i.e. (SX_reduced * weight_average_matrix) is: ", round((s/length(predicted_mtrx)) * 100, digits = 2), "%")   
  end 
=======
      a = 0.45 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced = reduce_size(adj_matrix ,x_matrix, y_matrix, skiped)
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(x_reduced, y_reduced, SX_reduced)
    elseif parameter == 2 
      SX_reduced, y_reduced, x_reduced = radomly_reduced_matrix(adj_matrix ,x_matrix, y_matrix, 0.7 ); 
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(x_reduced, y_reduced, SX_reduced)
    elseif parameter == 3
      a = 0.25 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced= to_drop(adj_matrix, x_matrix, y_matrix, skiped)
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(x_reduced, y_reduced, SX_reduced)
    end 
>>>>>>> teddy
  #return  accuracy, epochs, loss, predicted_mtrx
end 


<<<<<<< HEAD
 ############################################## generate data to test the my_function ##############################################


  #convert adj to S marix
  function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end

#This function makes data using Normal or Uniform distribution
function create_adjacency_matrix_normal_uniform(num, distro1)
  Random.seed!(10)
  ER_tmp = erdos_renyi( num , 10*(num) )
  BA_tmp = barabasi_albert( num , 8 )
  SF_tmp = static_scale_free( num , 8*(num) , 4 )
  WS_tmp = erdos_renyi( num , 10*(num) ) #barabasi_albert( NN_tmp , 5 )
  blocks_tmp = blockdiag( blockdiag( blockdiag(ER_tmp,BA_tmp),SF_tmp ), WS_tmp )
    
  #now add some edges between the blocks that are the communities
  for bb in 1:Int(round(num/10))
    for b1 in 0:3
      for b2 in 0:3
        if(b1 < b2)
          range1 = randperm(num)[1] + b1*num
          range2 = randperm(num)[1] + b2*num
          add_edge!( blocks_tmp , range1 , range2 )
        end
      end
    end
  end
  density_nn = Graphs.density(blocks_tmp)
  adj = Matrix(adjacency_matrix(blocks_tmp))

 
  d1 = rand(distro1(0, 1), 3 * num)
  d2 = rand(distro1(0, 1), 3 * num)
  d3 = rand(distro1(0, 1), 3 * num)
  c1 = Categorical( [0.5,0.25,0.25] )
  c2 = Categorical( [0.15,0.15,0.7] )
  c3 = Categorical( [0.5,0.5,0] )

 

   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1a = vcat( xd1 , xc1 )'
   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1b = vcat( xd1 , xc1 )'
   xd2 = rand( d2 ,  3 * num )
   xd2 = reshape(xd2, 3 ,num)
   xc2 = onehotbatch( rand( c2 , num ) , 1:3 )
   x2 = vcat( xd2 , xc2 )'
   xd3 = rand( d3 ,  3 * num )
   xd3 = reshape(xd3, 3 ,num)
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'

  X = vcat( x1a , x1b , x2 , x3 )
  y1a = onehotbatch( 1*ones(num) , 1:2 )'
  y1b = onehotbatch( 1*ones(num) , 1:2 )'
  y2 = onehotbatch( 2*ones(num) , 1:2 )'
  y3 = onehotbatch( 2*ones(num) , 1:2 )'
  Y = vcat(y1a, y1b, y2, y3)
  Y_to_use = vcat(1*ones(num), 1*ones(num), 2*ones(num), 2*ones(num))
  #println("size(X): ", size(X))
  #println("size(Y):", size(Y_to_use))
  return adj, X, Y_to_use
end 

#this function checks if I the y_hat data predicted by the network are the same with the y_hat_manually data, which is the data I predicted with the classical way: F(WX + bias)
#vec1 = y_hat
#vec2 = y_hat_manually

function check_validity(vec1, vec2)
  s = 0 
  for i in 1:length(vec1)
    if vec1[i] == vec2[i]
      s += 1
    end
  end
  if s == length(vec1)
    println("The two methods match")
  else
    println("Methods do not match")
  end 
end 

#This function loads the data and trains the neural network. 
#SX_ = the S*X matrix (X data)
#yhot = the Y data with (onehotbatch)


function get_adj_x_weight(num, k)
  #num = 1000
  distro1 = Normal 
  #k = 3
  ad, x, y = create_adjacency_matrix_normal_uniform(num, distro1)
  
  yhot = onehotbatch(y, [1, 2])
  S = A2S(ad)
  SX = S^k * x
  SX
  SX = SX'
  # println("SX = ", SX)
  train_x = SX
  # println("train_x = ", train_x)
  train_y = yhot
  #println("Training with raw data")
  #epochs, loss_on_train, loss_on_test,  model, resDict = load_and_train_3(SX, yhot, train_x, train_y)
  model, resDict, epochs, loss = load_and_train_model(SX, yhot)
  weight = resDict["model"].layers[1].weight
  # println("weight = ", weight)
  accuracy = round(mean( onecold( model(train_x), [1, 2] ) .== onecold(train_y, [1, 2]) ) * 100, digits = 2)
  println("Accuracy: ", accuracy, "%", ", k = $k, distribution = $distro1")      
  y_hat = onecold( model(train_x), [1, 2])
  y_actual = onecold(train_y, [1, 2])
  test = weight * SX
  #show(stdout, "text/plain", test)
  println("weight * SX = ", size(test))
  bias_ = resDict["model"].layers[1].bias
  bias_1 = fill(bias_[1], (1, size(test, 2)))
  bias_2 = fill(bias_[2], (1, size(test, 2)))
  bias = vcat(bias_1, bias_2)
  z = test + bias
  z_pred = softmax(z)
  y_hat_manually = onecold(softmax(z), [1, 2])
  #println("y_hat_manually = ", y_hat_manually)
  # check_validity(y_hat, y_hat_manually)
  SX_TH = onecold(test, [1, 2])
  #println("SX * Θ = ", SX_TH)
  #println("SX * Θ = ", y)
  #println(typeof(y))
  #show(stdout, "text/plain", test)
  y_ = Vector{Int64}()
  for i in 1:length(y)
    append!(y_, convert(Int64, y[i]))
  end  
  # println("SX * Θ_(no_onecold) = ", test)
  # println("SX * Θ  = ", SX_TH)
  # println("y_train = ", y_) #actual values
  s = 0
  for i in 1:length(y_)
    if y_[i] == SX_TH[i]
      s += 1
    end 
  end 
  ac = (s/length(y_))*100
  println("acccuracy SX * Θ = ", ac , "%")
  #show(stdout, "text/plain", S)
  #show(stdout, "text/plain", ad)
  #println(typeof(y))
  #println("weight * SX = ", test)
  return ad, x, weight, SX_TH
  end


  distro1 = Normal
  num = 1000
  d1 = rand(distro1(0, 1), 3 * num)
  d2 = rand(distro1(0, 1), 3 * num)
  d3 = rand(distro1(0, 1), 3 * num)
  c1 = Categorical( [0.5,0.25,0.25] )
  c2 = Categorical( [0.15,0.15,0.7] )
  c3 = Categorical( [0.5,0.5,0] )

 

   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1a = vcat( xd1 , xc1 )'
   xd1 = rand( d1 , 3 * num )
   xd1 = reshape(xd1, 3 ,num)
   xc1 = onehotbatch( rand( c1 , num ) , 1:3 )
   x1b = vcat( xd1 , xc1 )'
   xd2 = rand( d2 ,  3 * num )
   xd2 = reshape(xd2, 3 ,num)
   xc2 = onehotbatch( rand( c2 , num ) , 1:3 )
   x2 = vcat( xd2 , xc2 )'
   xd3 = rand( d3 ,  3 * num )
   xd3 = reshape(xd3, 3 ,num)
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'
   xc3 = onehotbatch( rand( c3 , num ) , 1:3 )
   x3 = vcat( xd3 , xc3 )'

  X = vcat( x1a , x1b , x2 , x3 )
  y1a = onehotbatch( 1*ones(num) , 1:2 )'
  y1b = onehotbatch( 1*ones(num) , 1:2 )'
  y2 = onehotbatch( 2*ones(num) , 1:2 )'
  y3 = onehotbatch( 2*ones(num) , 1:2 )'
  Y = vcat(y1a, y1b, y2, y3)


  Y_to_use = vcat(1*ones(num), 1*ones(num), 2*ones(num), 2*ones(num))

  adj_mtrx, x_mtrx, weight_mtrx, sx_theta = get_adj_x_weight(1000, 3)



my_function(adj_mtrx, x_mtrx, Y_to_use, 1);
=======
for i in 1:3
  my_function(adj_mtrx, x_mtrx, Y_to_use, i)
  println("------------------------------------------------------------------------------------------------------------------")
end 
>>>>>>> teddy
