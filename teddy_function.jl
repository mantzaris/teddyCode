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
using MLDatasets: Cora


 ############################################## generate data to test the my_function ##############################################


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
    x_reduced_ = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    y_reduced_ = y[Not(x_dim - n_col_row_to_reduce + 1: end), :]
    SX_reduced_ = s_reduced^k * x_reduced_
    return SX_reduced_, y_reduced_, x_reduced_
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
    x_reduced_ = x_mtrx
    y_reduced_ = y_mtrx
    return SX_reduced_, y_reduced_, x_reduced_
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
  return SX_reduced, Ydata_, X_
end 




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


  ########################### my_function modified ###########################

  #adj_matrix = adjacency matrix
  #x_matrix = x matrix
  #y_matrix = y matrix (actual data)
  #paremeter = says which method will be used to split the adj_matrix 
  
  function my_function(adj_matrix, x_matrix, y_matrix, parameter)
    #get Y' data by training the network, before seting the parameter
    k = 3
    S = A2S(adj_matrix)
    SX = S^k * x_matrix
    if parameter == 1
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
      #p = 0.3
      #dropping = Int(size(adj_matrix, 1) * p)
      a = 0.25 #droping the a% of the matrix
      skiped = round(size(adj_matrix, 1) * a , digits = 0)
      SX_reduced, y_reduced, x_reduced= to_drop(adj_matrix, x_matrix, y_matrix, skiped)
      y_reduced = vec(y_reduced)
      load_and_train_model_one_shot_learining(x_reduced, y_reduced, SX_reduced)
    end 
  #return  accuracy, epochs, loss, predicted_mtrx
end 




for i in 1:3
  my_function(adj_mtrx, x_mtrx, Y_to_use, i)
  println("------------------------------------------------------------------------------------------------------------------")
end 
